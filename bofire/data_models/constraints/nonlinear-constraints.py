"""
Nonlinear constraint implementation for BoFire.

This module provides support for arbitrary nonlinear constraints in Bayesian optimization,
extending beyond BoFire's native linear constraint support.
"""

import functools
from typing import Callable, List, Optional, Union, Dict, Any
import warnings

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import Input


class NonlinearConstraint(Constraint):
    """
    Arbitrary nonlinear constraint for Bayesian optimization.
    
    Supports inequality constraints (g(x) <= 0) and equality constraints (h(x) = 0).
    Constraints are defined using Python expressions that reference DataFrame rows.
    
    The constraint is considered **feasible** when:
    - Inequality: g(x) <= tol (default tol=0)
    - Equality: |h(x)| <= tol (default tol=1e-6)
    
    Parameters
    ----------
    features : List[str]
        List of input feature keys that appear in the constraint expression.
        These must match keys in the Domain's Inputs.
    constraint_name : str
        Unique human-readable identifier for the constraint.
        Used in logging, debugging, and constraint violation reports.
    constraint_expr : str
        Python expression defining the constraint function.
        Must reference features using `row['feature_name']` syntax.
        Should return a float where values <= 0 indicate feasibility.
        
        Supported functions: np.* (numpy), abs, max, min, sqrt, exp, log
        
        Examples:
        - "row['Temperature'] + 100 * row['Pd_fraction']**2 - 350"
        - "row['x1']**2 + row['x2']**2 - 1.0"  # Circle constraint
        - "0 if row['T'] <= 280 else max(0.3 - row['Pd'], row['Pd'] - 0.6)"
    constraint_type : str, default="inequality"
        Type of constraint: "inequality" (g(x) <= 0) or "equality" (h(x) = 0)
    tolerance : float, default=1e-6
        Numerical tolerance for constraint satisfaction.
        For inequality: g(x) <= tolerance is feasible
        For equality: |h(x)| <= tolerance is feasible
    
    Examples
    --------
    >>> # Sintering temperature limit: T + 100*Pd^2 <= 350
    >>> constraint = NonlinearConstraint(
    ...     features=["Temperature", "Pd_fraction"],
    ...     constraint_name="Sintering Limit",
    ...     constraint_expr="row['Temperature'] + 100 * row['Pd_fraction']**2 - 350",
    ...     constraint_type="inequality"
    ... )
    
    >>> # Circular feasible region: x1^2 + x2^2 <= 1
    >>> constraint = NonlinearConstraint(
    ...     features=["x1", "x2"],
    ...     constraint_name="Circle",
    ...     constraint_expr="row['x1']**2 + row['x2']**2 - 1.0",
    ...     constraint_type="inequality"
    ... )
    
    >>> # Conditional constraint: if T>280, then Pd in [0.3, 0.6]
    >>> constraint = NonlinearConstraint(
    ...     features=["Temperature", "Pd_fraction"],
    ...     constraint_name="Selectivity Window",
    ...     constraint_expr=(
    ...         "0 if row['Temperature'] <= 280 else "
    ...         "max(0.3 - row['Pd_fraction'], row['Pd_fraction'] - 0.6)"
    ...     ),
    ...     constraint_type="inequality"
    ... )
    
    Notes
    -----
    - Expression evaluation uses safe globals (no __builtins__ access)
    - Compiled functions are cached for performance (via @cached_property)
    - Jacobians computed via finite differences (automatic differentiation planned)
    - For performance-critical applications, consider providing analytical Jacobians
    
    Warnings
    --------
    - Expressions use eval() with restricted namespace (only math functions allowed)
    - Division by zero, NaN, or inf in expressions will raise errors
    - Complex constraints may slow down candidate generation
    
    See Also
    --------
    LinearInequalityConstraint : For linear constraints (more efficient)
    LinearEqualityConstraint : For linear equality constraints
    """
    
    type: str = Field(
        default="NonlinearConstraint",
        description="Discriminator for Pydantic unions"
    )
    
    # Core attributes
    features: List[str] = Field(
        ...,
        min_length=1,
        description="Input feature keys involved in constraint"
    )
    constraint_name: str = Field(
        ...,
        min_length=1,
        description="Unique constraint identifier"
    )
    constraint_expr: str = Field(
        ...,
        min_length=1,
        description="Python expression for constraint (e.g., \"row['x1']**2 + row['x2']**2 - 1\")"
    )
    constraint_type: str = Field(
        default="inequality",
        description="Constraint type: 'inequality' (g(x)<=0) or 'equality' (h(x)=0)"
    )
    tolerance: float = Field(
        default=1e-6,
        gt=0,
        description="Numerical tolerance for constraint satisfaction"
    )
    
    # Metadata
    description: Optional[str] = Field(
        default=None,
        description="Optional detailed description of constraint purpose"
    )
    
    @field_validator('constraint_type')
    @classmethod
    def validate_constraint_type(cls, v: str) -> str:
        """Validate constraint type is one of the allowed values."""
        allowed = {'inequality', 'equality'}
        if v not in allowed:
            raise ValueError(
                f"constraint_type must be one of {allowed}, got '{v}'"
            )
        return v
    
    @field_validator('constraint_expr')
    @classmethod
    def validate_expression_syntax(cls, v: str) -> str:
        """
        Validate that the constraint expression is syntactically valid Python.
        
        This performs a basic syntax check but does not execute the expression.
        """
        import ast
        try:
            # Check if it's valid Python syntax
            ast.parse(v, mode='eval')
        except SyntaxError as e:
            raise ValueError(
                f"Invalid Python expression syntax: {e}\n"
                f"Expression: {v}"
            )
        return v
    
    @model_validator(mode='after')
    def validate_features_in_expression(self) -> 'NonlinearConstraint':
        """
        Warn if features listed in 'features' don't appear in the expression.
        
        This is a soft warning (not an error) since expressions might use
        different row access patterns.
        """
        expr = self.constraint_expr
        missing_features = [
            f for f in self.features 
            if f"row['{f}']" not in expr and f'row["{f}"]' not in expr
        ]
        
        if missing_features:
            warnings.warn(
                f"Constraint '{self.constraint_name}': Features {missing_features} "
                f"are listed but don't appear in expression. "
                f"Expression: {expr}",
                UserWarning
            )
        
        return self
    
    @functools.cached_property
    def _compiled_func(self) -> Callable:
        """
        Compile the constraint expression into a callable function.
        
        The function is cached to avoid recompilation on every evaluation.
        Uses a restricted namespace for safety (no access to __builtins__).
        
        Returns
        -------
        Callable
            Function that takes a DataFrame row and returns constraint value
        
        Raises
        ------
        ValueError
            If expression cannot be compiled or references unsafe functions
        """
        # Define safe namespace (only math functions, no file I/O, no imports)
        safe_globals = {
            # NumPy functions
            'np': np,
            'array': np.array,
            'sqrt': np.sqrt,
            'exp': np.exp,
            'log': np.log,
            'log10': np.log10,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'arcsin': np.arcsin,
            'arccos': np.arccos,
            'arctan': np.arctan,
            'sinh': np.sinh,
            'cosh': np.cosh,
            'tanh': np.tanh,
            
            # Python builtins (safe subset)
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'round': round,
            'pow': pow,
            
            # Explicitly block dangerous functions
            '__builtins__': {},
        }
        
        try:
            # Compile expression as a lambda function
            func_code = f"lambda row: {self.constraint_expr}"
            compiled_func = eval(func_code, safe_globals, {})
            
            # Test compilation with a dummy row (catch some errors early)
            dummy_row = pd.Series({f: 0.0 for f in self.features})
            try:
                result = compiled_func(dummy_row)
                if not isinstance(result, (int, float, np.number)):
                    raise TypeError(
                        f"Expression must return a number, got {type(result)}"
                    )
            except Exception as e:
                warnings.warn(
                    f"Constraint '{self.constraint_name}' test evaluation failed: {e}. "
                    f"This may be normal for conditional expressions.",
                    UserWarning
                )
            
            return compiled_func
            
        except SyntaxError as e:
            raise ValueError(
                f"Failed to compile constraint expression: {e}\n"
                f"Expression: {self.constraint_expr}"
            )
        except Exception as e:
            raise ValueError(
                f"Error compiling constraint '{self.constraint_name}': {e}\n"
                f"Expression: {self.constraint_expr}"
            )
    
    def __call__(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """
        Primary BoFire interface: evaluate constraint and return as DataFrame.
        
        This method is called by BoFire strategies to check constraint satisfaction.
        
        Parameters
        ----------
        experiments : pd.DataFrame
            Experimental conditions to evaluate
        
        Returns
        -------
        pd.DataFrame
            Single-column DataFrame with constraint values.
            Column name is the constraint_name.
            Values <= 0 indicate feasibility (for inequality constraints).
        
        Examples
        --------
        >>> constraint = NonlinearConstraint(...)
        >>> experiments = pd.DataFrame({'x1': [0.5, 1.5], 'x2': [0.3, 0.8]})
        >>> result = constraint(experiments)
        >>> print(result)
           Circle
        0  -0.66
        1   1.89
        """
        constraint_values = self.evaluate(experiments)
        return pd.DataFrame({self.constraint_name: constraint_values})
    
    def evaluate(self, experiments: pd.DataFrame) -> pd.Series:
        """
        Evaluate constraint function values on experiments.
        
        Parameters
        ----------
        experiments : pd.DataFrame
            Experimental conditions. Must contain all features in self.features.
        
        Returns
        -------
        pd.Series
            Constraint values for each experiment.
            For inequality: values <= 0 indicate feasibility
            For equality: values = 0 indicate feasibility (check with is_fulfilled)
        
        Raises
        ------
        ValueError
            If required features are missing from experiments
        RuntimeError
            If expression evaluation fails
        
        Examples
        --------
        >>> values = constraint.evaluate(experiments)
        >>> print(values)
        0   -0.66
        1    1.89
        dtype: float64
        """
        # Validate features exist
        missing_features = set(self.features) - set(experiments.columns)
        if missing_features:
            raise ValueError(
                f"Constraint '{self.constraint_name}' requires features "
                f"{missing_features} which are not in the DataFrame.\n"
                f"Available columns: {list(experiments.columns)}"
            )
        
        func = self._compiled_func
        
        try:
            # Apply function row-by-row
            constraint_values = experiments.apply(func, axis=1)
            
            # Validate output
            if not pd.api.types.is_numeric_dtype(constraint_values):
                raise TypeError(
                    f"Constraint expression must return numeric values, "
                    f"got dtype {constraint_values.dtype}"
                )
            
            # Check for NaN or inf
            if constraint_values.isna().any():
                nan_indices = constraint_values[constraint_values.isna()].index.tolist()
                raise RuntimeError(
                    f"Constraint '{self.constraint_name}' produced NaN values "
                    f"for rows {nan_indices[:5]}..."  # Show first 5
                )
            
            if np.isinf(constraint_values).any():
                inf_indices = constraint_values[np.isinf(constraint_values)].index.tolist()
                raise RuntimeError(
                    f"Constraint '{self.constraint_name}' produced inf values "
                    f"for rows {inf_indices[:5]}..."
                )
            
            return constraint_values
            
        except Exception as e:
            # Provide context for debugging
            sample_row = experiments.iloc[0].to_dict() if len(experiments) > 0 else {}
            raise RuntimeError(
                f"Error evaluating constraint '{self.constraint_name}':\n"
                f"  Expression: {self.constraint_expr}\n"
                f"  Error: {e}\n"
                f"  Sample row: {sample_row}"
            ) from e
    
    def is_fulfilled(
        self, 
        experiments: pd.DataFrame, 
        tol: Optional[float] = None
    ) -> pd.Series:
        """
        Check if experiments satisfy the constraint.
        
        Parameters
        ----------
        experiments : pd.DataFrame
            Experimental conditions to check
        tol : float, optional
            Tolerance override. If None, uses self.tolerance.
        
        Returns
        -------
        pd.Series
            Boolean series where True indicates constraint is satisfied
        
        Examples
        --------
        >>> is_feasible = constraint.is_fulfilled(experiments)
        >>> print(is_feasible)
        0     True
        1    False
        dtype: bool
        
        >>> # Check which experiments are infeasible
        >>> infeasible = experiments[~is_feasible]
        """
        if tol is None:
            tol = self.tolerance
        
        constraint_values = self.evaluate(experiments)
        
        if self.constraint_type == 'inequality':
            # g(x) <= tol is feasible
            return constraint_values <= tol
        else:  # equality
            # |h(x)| <= tol is feasible
            return np.abs(constraint_values) <= tol
    
    def jacobian(
        self, 
        experiments: pd.DataFrame, 
        eps: float = 1e-8
    ) -> pd.DataFrame:
        """
        Compute numerical gradient of constraint via finite differences.
        
        Useful for gradient-based acquisition function optimization.
        Uses vectorized computation for performance.
        
        Parameters
        ----------
        experiments : pd.DataFrame
            Points at which to compute gradients
        eps : float, default=1e-8
            Finite difference step size. Should be sqrt(machine epsilon)
            for central differences, but we use forward differences here.
        
        Returns
        -------
        pd.DataFrame
            Gradients with columns for each feature in self.features.
            Shape: (n_experiments, n_features)
        
        Notes
        -----
        This uses forward finite differences: dg/dx ≈ (g(x+eps) - g(x)) / eps
        For better accuracy, consider central differences or automatic differentiation.
        
        Examples
        --------
        >>> grads = constraint.jacobian(experiments)
        >>> print(grads)
              x1        x2
        0  1.000  0.600
        1  3.000  1.600
        """
        func = self._compiled_func
        
        # Base function values (vectorized)
        base_values = self.evaluate(experiments).values
        
        # Compute gradient for each feature (vectorized)
        gradients = {}
        for feature in self.features:
            # Perturb all rows at once
            perturbed = experiments.copy()
            perturbed[feature] = perturbed[feature] + eps
            
            # Evaluate on perturbed data
            perturbed_values = self.evaluate(perturbed).values
            
            # Finite difference
            gradients[feature] = (perturbed_values - base_values) / eps
        
        return pd.DataFrame(gradients, index=experiments.index)
    
    def validate_inputs(self, inputs: Union[List[Input], Any]) -> None:
        """
        Validate that all required features exist in domain inputs.
        
        Should be called by Domain during initialization.
        
        Parameters
        ----------
        inputs : Inputs or list of Input
            Input features from Domain
        
        Raises
        ------
        ValueError
            If required features are missing from inputs
        
        Examples
        --------
        >>> from bofire.data_models.domain.api import Domain
        >>> domain = Domain(inputs=inputs, outputs=outputs, constraints=[constraint])
        >>> # validate_inputs is called automatically
        """
        # Handle both Inputs object and list of Input features
        if hasattr(inputs, 'features'):
            input_keys = {f.key for f in inputs.features}
        elif isinstance(inputs, list):
            input_keys = {f.key for f in inputs}
        else:
            input_keys = {f.key for f in inputs.get_features(Input)}
        
        missing_features = set(self.features) - input_keys
        
        if missing_features:
            raise ValueError(
                f"NonlinearConstraint '{self.constraint_name}' requires features "
                f"{sorted(missing_features)} which are not present in domain inputs.\n"
                f"Available input features: {sorted(input_keys)}"
            )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"NonlinearConstraint(name='{self.constraint_name}', "
            f"type='{self.constraint_type}', "
            f"features={self.features})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"NonlinearConstraint(\n"
            f"  name='{self.constraint_name}',\n"
            f"  type='{self.constraint_type}',\n"
            f"  features={self.features},\n"
            f"  expr='{self.constraint_expr}',\n"
            f"  tolerance={self.tolerance}\n"
            f")"
        )


# =============================================================================
# Helper Functions
# =============================================================================

def filter_candidates_by_constraints(
    candidates: pd.DataFrame,
    constraints: List[NonlinearConstraint],
    tol: Optional[float] = None
) -> pd.DataFrame:
    """
    Filter candidates to only include those satisfying all constraints.
    
    This is a utility function for post-generation constraint filtering.
    Can be used by any BoFire strategy.
    
    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate experiments to filter
    constraints : List[NonlinearConstraint]
        Constraints to enforce
    tol : float, optional
        Tolerance for constraint satisfaction. If None, uses each constraint's
        default tolerance.
    
    Returns
    -------
    pd.DataFrame
        Filtered candidates (may be empty if none are feasible)
    
    Examples
    --------
    >>> from bofire.strategies.api import MoboStrategy
    >>> strategy = MoboStrategy(...)
    >>> raw_candidates = strategy.ask(candidate_count=10)
    >>> feasible = filter_candidates_by_constraints(raw_candidates, constraints)
    >>> print(f"Feasibility rate: {len(feasible)/len(raw_candidates):.1%}")
    """
    if not constraints:
        return candidates
    
    # Start with all candidates being feasible
    feasible_mask = pd.Series([True] * len(candidates), index=candidates.index)
    
    # Apply each constraint (AND logic)
    for constraint in constraints:
        feasible_mask &= constraint.is_fulfilled(candidates, tol=tol)
    
    # Return only feasible candidates
    feasible_candidates = candidates[feasible_mask]
    
    return feasible_candidates


def get_constraint_violations(
    experiments: pd.DataFrame,
    constraints: List[NonlinearConstraint]
) -> pd.DataFrame:
    """
    Compute constraint violation amounts for each experiment.
    
    Useful for debugging and understanding which constraints are active.
    
    Parameters
    ----------
    experiments : pd.DataFrame
        Experiments to check
    constraints : List[NonlinearConstraint]
        Constraints to evaluate
    
    Returns
    -------
    pd.DataFrame
        Violation amounts for each constraint (columns) and experiment (rows).
        Positive values indicate violations, negative values indicate slack.
    
    Examples
    --------
    >>> violations = get_constraint_violations(experiments, constraints)
    >>> print(violations)
       Sintering Limit  Selectivity Window
    0            -50.2                 0.0
    1             12.5                 0.1  # Violates both!
    2            -23.1                 0.0
    """
    violation_data = {}
    
    for constraint in constraints:
        values = constraint.evaluate(experiments)
        
        if constraint.constraint_type == 'inequality':
            # For inequality g(x) <= 0: violation = max(0, g(x))
            violations = values.clip(lower=0)
        else:  # equality
            # For equality h(x) = 0: violation = |h(x)|
            violations = values.abs()
        
        violation_data[constraint.constraint_name] = violations
    
    return pd.DataFrame(violation_data, index=experiments.index)
