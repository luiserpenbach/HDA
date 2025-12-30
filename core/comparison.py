"""
Data Comparison and Regression (P2)
===================================
Tools for comparing tests and performing regression analysis:
- Test-to-test comparison
- Golden reference comparison
- Multi-parameter regression
- Correlation analysis
- Deviation tracking
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime


@dataclass
class ComparisonResult:
    """Result of comparing two values."""
    parameter: str
    value_a: float
    value_b: float
    difference: float
    percent_difference: float
    within_tolerance: bool
    tolerance: float
    unit: str = ""
    
    @property
    def direction(self) -> str:
        if self.difference > 0:
            return "higher"
        elif self.difference < 0:
            return "lower"
        return "equal"


@dataclass
class TestComparison:
    """Complete comparison between two tests."""
    test_a_id: str
    test_b_id: str
    comparisons: List[ComparisonResult]
    overall_pass: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def n_parameters(self) -> int:
        return len(self.comparisons)
    
    @property
    def n_within_tolerance(self) -> int:
        return sum(1 for c in self.comparisons if c.within_tolerance)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame([
            {
                'Parameter': c.parameter,
                f'{self.test_a_id}': f"{c.value_a:.4g}",
                f'{self.test_b_id}': f"{c.value_b:.4g}",
                'Difference': f"{c.difference:+.4g}",
                'Δ%': f"{c.percent_difference:+.2f}%",
                'Tolerance': f"±{c.tolerance:.2f}%",
                'Status': '✓' if c.within_tolerance else '✗',
            }
            for c in self.comparisons
        ])
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"Comparison: {self.test_a_id} vs {self.test_b_id}",
            f"Result: {'PASS' if self.overall_pass else 'FAIL'}",
            f"Parameters within tolerance: {self.n_within_tolerance}/{self.n_parameters}",
            "",
        ]
        
        for c in self.comparisons:
            status = "✓" if c.within_tolerance else "✗"
            lines.append(
                f"  {status} {c.parameter}: {c.percent_difference:+.2f}% "
                f"(tolerance: ±{c.tolerance:.1f}%)"
            )
        
        return "\n".join(lines)


@dataclass  
class RegressionResult:
    """Result of linear regression analysis."""
    x_parameter: str
    y_parameter: str
    slope: float
    intercept: float
    r_squared: float
    std_error: float
    n_points: int
    p_value: Optional[float] = None
    confidence_interval_slope: Optional[Tuple[float, float]] = None
    prediction_equation: str = ""
    
    def predict(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Predict y from x."""
        return self.slope * x + self.intercept
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"Regression: {self.y_parameter} = f({self.x_parameter})",
            f"  Equation: y = {self.slope:.4g}x + {self.intercept:.4g}",
            f"  R² = {self.r_squared:.4f}",
            f"  Standard Error = {self.std_error:.4g}",
            f"  N = {self.n_points}",
        ]
        
        if self.confidence_interval_slope:
            lines.append(f"  Slope 95% CI: [{self.confidence_interval_slope[0]:.4g}, "
                        f"{self.confidence_interval_slope[1]:.4g}]")
        
        return "\n".join(lines)


@dataclass
class CorrelationMatrix:
    """Correlation analysis results."""
    parameters: List[str]
    matrix: np.ndarray
    p_values: Optional[np.ndarray] = None
    
    def get_correlation(self, param1: str, param2: str) -> float:
        """Get correlation between two parameters."""
        i = self.parameters.index(param1)
        j = self.parameters.index(param2)
        return self.matrix[i, j]
    
    def get_strong_correlations(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Get parameter pairs with strong correlation."""
        strong = []
        n = len(self.parameters)
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.matrix[i, j]
                if abs(corr) >= threshold:
                    strong.append((self.parameters[i], self.parameters[j], corr))
        return sorted(strong, key=lambda x: -abs(x[2]))
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame(
            self.matrix,
            index=self.parameters,
            columns=self.parameters
        )


@dataclass
class GoldenReference:
    """Golden reference data for comparison."""
    name: str
    parameters: Dict[str, float]
    tolerances: Dict[str, float]  # Parameter -> % tolerance
    uncertainties: Optional[Dict[str, float]] = None
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    source_tests: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'tolerances': self.tolerances,
            'uncertainties': self.uncertainties,
            'created_date': self.created_date,
            'description': self.description,
            'source_tests': self.source_tests,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldenReference':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            parameters=data['parameters'],
            tolerances=data['tolerances'],
            uncertainties=data.get('uncertainties'),
            created_date=data.get('created_date', datetime.now().isoformat()),
            description=data.get('description', ''),
            source_tests=data.get('source_tests', []),
        )


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compare_values(
    value_a: float,
    value_b: float,
    parameter: str,
    tolerance_percent: float = 5.0,
    unit: str = "",
) -> ComparisonResult:
    """Compare two values and check if within tolerance."""
    difference = value_a - value_b
    
    # Calculate percent difference relative to average
    avg = (abs(value_a) + abs(value_b)) / 2
    if avg > 1e-10:
        percent_diff = (difference / avg) * 100
    else:
        percent_diff = 0.0 if abs(difference) < 1e-10 else float('inf')
    
    within_tol = abs(percent_diff) <= tolerance_percent
    
    return ComparisonResult(
        parameter=parameter,
        value_a=value_a,
        value_b=value_b,
        difference=difference,
        percent_difference=percent_diff,
        within_tolerance=within_tol,
        tolerance=tolerance_percent,
        unit=unit,
    )


def compare_tests(
    test_a: Dict[str, float],
    test_b: Dict[str, float],
    test_a_id: str,
    test_b_id: str,
    tolerances: Optional[Dict[str, float]] = None,
    default_tolerance: float = 5.0,
    parameters: Optional[List[str]] = None,
) -> TestComparison:
    """
    Compare two tests across multiple parameters.
    
    Args:
        test_a: Dict of parameter -> value for test A
        test_b: Dict of parameter -> value for test B
        test_a_id: Identifier for test A
        test_b_id: Identifier for test B
        tolerances: Dict of parameter -> tolerance %
        default_tolerance: Default tolerance if not specified
        parameters: List of parameters to compare (None = all common)
        
    Returns:
        TestComparison result
    """
    tolerances = tolerances or {}
    
    # Determine parameters to compare
    if parameters is None:
        parameters = list(set(test_a.keys()) & set(test_b.keys()))
    
    comparisons = []
    all_pass = True
    
    for param in parameters:
        if param not in test_a or param not in test_b:
            continue
        
        val_a = test_a[param]
        val_b = test_b[param]
        
        # Skip if not numeric
        if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
            continue
        
        tol = tolerances.get(param, default_tolerance)
        
        result = compare_values(val_a, val_b, param, tol)
        comparisons.append(result)
        
        if not result.within_tolerance:
            all_pass = False
    
    return TestComparison(
        test_a_id=test_a_id,
        test_b_id=test_b_id,
        comparisons=comparisons,
        overall_pass=all_pass,
    )


def compare_to_golden(
    test_data: Dict[str, float],
    test_id: str,
    golden: GoldenReference,
) -> TestComparison:
    """Compare test results to golden reference."""
    return compare_tests(
        test_a=test_data,
        test_b=golden.parameters,
        test_a_id=test_id,
        test_b_id=f"Golden ({golden.name})",
        tolerances=golden.tolerances,
    )


def create_golden_from_campaign(
    df: pd.DataFrame,
    name: str,
    parameters: List[str],
    tolerance_multiplier: float = 3.0,
    method: str = 'mean',
    min_tests: int = 5,
) -> GoldenReference:
    """
    Create golden reference from campaign data.
    
    Args:
        df: Campaign DataFrame
        name: Name for the golden reference
        parameters: Parameters to include
        tolerance_multiplier: Multiply std dev by this for tolerance
        method: 'mean' or 'median' for central value
        min_tests: Minimum tests required
        
    Returns:
        GoldenReference
    """
    if len(df) < min_tests:
        raise ValueError(f"Need at least {min_tests} tests, got {len(df)}")
    
    ref_values = {}
    tolerances = {}
    uncertainties = {}
    
    for param in parameters:
        if param not in df.columns:
            continue
        
        values = df[param].dropna()
        if len(values) < min_tests:
            continue
        
        if method == 'mean':
            central = values.mean()
        else:
            central = values.median()
        
        std = values.std()
        
        ref_values[param] = float(central)
        
        # Tolerance as percentage
        if abs(central) > 1e-10:
            tol_pct = (tolerance_multiplier * std / abs(central)) * 100
        else:
            tol_pct = 10.0  # Default
        
        tolerances[param] = float(max(tol_pct, 1.0))  # Minimum 1%
        uncertainties[param] = float(std)
    
    source_tests = df['test_id'].tolist() if 'test_id' in df.columns else []
    
    return GoldenReference(
        name=name,
        parameters=ref_values,
        tolerances=tolerances,
        uncertainties=uncertainties,
        source_tests=source_tests,
        description=f"Generated from {len(df)} tests using {method}",
    )


# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

def linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    x_name: str = "x",
    y_name: str = "y",
) -> RegressionResult:
    """
    Perform linear regression.
    
    Args:
        x: Independent variable values
        y: Dependent variable values
        x_name: Name of x parameter
        y_name: Name of y parameter
        
    Returns:
        RegressionResult
    """
    # Remove NaN pairs
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    n = len(x_valid)
    if n < 3:
        raise ValueError("Need at least 3 valid points for regression")
    
    # Calculate regression
    x_mean = np.mean(x_valid)
    y_mean = np.mean(y_valid)
    
    ss_xx = np.sum((x_valid - x_mean) ** 2)
    ss_yy = np.sum((y_valid - y_mean) ** 2)
    ss_xy = np.sum((x_valid - x_mean) * (y_valid - y_mean))
    
    if ss_xx < 1e-10:
        raise ValueError("X values have no variance")
    
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    
    # Predictions and residuals
    y_pred = slope * x_valid + intercept
    residuals = y_valid - y_pred
    ss_res = np.sum(residuals ** 2)
    
    # R-squared
    if ss_yy > 1e-10:
        r_squared = 1 - ss_res / ss_yy
    else:
        r_squared = 1.0
    
    # Standard error
    if n > 2:
        mse = ss_res / (n - 2)
        std_error = np.sqrt(mse)
        
        # Slope standard error
        slope_se = std_error / np.sqrt(ss_xx)
        
        # 95% confidence interval for slope (t-distribution approximation)
        t_val = 2.0  # Approximate for large n
        ci_low = slope - t_val * slope_se
        ci_high = slope + t_val * slope_se
    else:
        std_error = 0.0
        ci_low, ci_high = slope, slope
    
    # Create equation string
    sign = "+" if intercept >= 0 else "-"
    equation = f"{y_name} = {slope:.4g}·{x_name} {sign} {abs(intercept):.4g}"
    
    return RegressionResult(
        x_parameter=x_name,
        y_parameter=y_name,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        std_error=float(std_error),
        n_points=n,
        confidence_interval_slope=(float(ci_low), float(ci_high)),
        prediction_equation=equation,
    )


def multi_regression(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
) -> Dict[str, RegressionResult]:
    """
    Perform regression of y against multiple x variables.
    
    Returns dict of x_col -> RegressionResult
    """
    results = {}
    
    y = df[y_col].values
    
    for x_col in x_cols:
        if x_col not in df.columns:
            continue
        
        x = df[x_col].values
        
        try:
            result = linear_regression(x, y, x_col, y_col)
            results[x_col] = result
        except ValueError:
            continue
    
    return results


# =============================================================================
# CORRELATION FUNCTIONS
# =============================================================================

def calculate_correlation_matrix(
    df: pd.DataFrame,
    parameters: Optional[List[str]] = None,
) -> CorrelationMatrix:
    """
    Calculate correlation matrix for parameters.
    
    Args:
        df: DataFrame with data
        parameters: List of columns (None = all numeric)
        
    Returns:
        CorrelationMatrix
    """
    if parameters is None:
        parameters = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    
    # Filter to available columns
    parameters = [p for p in parameters if p in df.columns]
    
    if len(parameters) < 2:
        raise ValueError("Need at least 2 parameters for correlation")
    
    # Calculate correlation matrix
    subset = df[parameters].dropna()
    matrix = subset.corr().values
    
    return CorrelationMatrix(
        parameters=parameters,
        matrix=matrix,
    )


def find_correlations(
    df: pd.DataFrame,
    parameters: Optional[List[str]] = None,
    threshold: float = 0.7,
) -> List[Tuple[str, str, float]]:
    """Find strongly correlated parameter pairs."""
    corr_matrix = calculate_correlation_matrix(df, parameters)
    return corr_matrix.get_strong_correlations(threshold)


# =============================================================================
# DEVIATION TRACKING
# =============================================================================

@dataclass
class DeviationTracker:
    """Track deviations from expected values over time."""
    parameter: str
    expected_value: float
    tolerance_percent: float
    history: List[Tuple[str, float, float]] = field(default_factory=list)  # (test_id, value, deviation_pct)
    
    def add_measurement(self, test_id: str, value: float):
        """Add a measurement and calculate deviation."""
        if abs(self.expected_value) > 1e-10:
            deviation_pct = ((value - self.expected_value) / self.expected_value) * 100
        else:
            deviation_pct = 0.0
        
        self.history.append((test_id, value, deviation_pct))
    
    @property
    def n_measurements(self) -> int:
        return len(self.history)
    
    @property
    def n_out_of_tolerance(self) -> int:
        return sum(1 for _, _, dev in self.history if abs(dev) > self.tolerance_percent)
    
    @property
    def mean_deviation(self) -> float:
        if not self.history:
            return 0.0
        return np.mean([dev for _, _, dev in self.history])
    
    @property
    def max_deviation(self) -> float:
        if not self.history:
            return 0.0
        return max(abs(dev) for _, _, dev in self.history)
    
    def get_trend(self) -> Optional[Tuple[str, float]]:
        """Check for trend in deviations."""
        if len(self.history) < 5:
            return None
        
        deviations = np.array([dev for _, _, dev in self.history])
        x = np.arange(len(deviations))
        
        # Simple linear regression
        slope = np.polyfit(x, deviations, 1)[0]
        
        # Normalize by range
        dev_range = np.ptp(deviations)
        if dev_range > 0:
            norm_slope = slope * len(deviations) / dev_range
        else:
            norm_slope = 0
        
        if abs(norm_slope) > 0.3:
            direction = "increasing" if slope > 0 else "decreasing"
            return (direction, float(slope))
        
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame."""
        return pd.DataFrame([
            {
                'test_id': test_id,
                'value': value,
                'expected': self.expected_value,
                'deviation_pct': deviation,
                'in_tolerance': abs(deviation) <= self.tolerance_percent,
            }
            for test_id, value, deviation in self.history
        ])


def track_deviations(
    df: pd.DataFrame,
    parameter: str,
    expected_value: float,
    tolerance_percent: float = 5.0,
    test_id_col: str = 'test_id',
) -> DeviationTracker:
    """
    Track deviations of a parameter from expected value.
    
    Args:
        df: Campaign DataFrame
        parameter: Column name to track
        expected_value: Expected/nominal value
        tolerance_percent: Tolerance in percent
        test_id_col: Column with test IDs
        
    Returns:
        DeviationTracker with history
    """
    tracker = DeviationTracker(
        parameter=parameter,
        expected_value=expected_value,
        tolerance_percent=tolerance_percent,
    )
    
    for _, row in df.iterrows():
        test_id = row.get(test_id_col, f"row_{_}")
        value = row.get(parameter)
        
        if pd.notna(value):
            tracker.add_measurement(str(test_id), float(value))
    
    return tracker


# =============================================================================
# CAMPAIGN COMPARISON
# =============================================================================

def compare_campaigns(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    campaign_a_name: str,
    campaign_b_name: str,
    parameters: List[str],
    tolerances: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compare statistics between two campaigns.
    
    Returns:
        Dict with comparison results
    """
    tolerances = tolerances or {}
    results = {
        'campaign_a': campaign_a_name,
        'campaign_b': campaign_b_name,
        'n_tests_a': len(df_a),
        'n_tests_b': len(df_b),
        'parameters': {},
    }
    
    for param in parameters:
        if param not in df_a.columns or param not in df_b.columns:
            continue
        
        vals_a = df_a[param].dropna()
        vals_b = df_b[param].dropna()
        
        if len(vals_a) < 2 or len(vals_b) < 2:
            continue
        
        mean_a = vals_a.mean()
        mean_b = vals_b.mean()
        std_a = vals_a.std()
        std_b = vals_b.std()
        
        # Percent difference in means
        avg_mean = (abs(mean_a) + abs(mean_b)) / 2
        if avg_mean > 1e-10:
            mean_diff_pct = ((mean_a - mean_b) / avg_mean) * 100
        else:
            mean_diff_pct = 0.0
        
        # F-test for variance
        if std_b > 1e-10:
            f_ratio = (std_a / std_b) ** 2
        else:
            f_ratio = float('inf')
        
        tol = tolerances.get(param, 5.0)
        
        results['parameters'][param] = {
            'mean_a': float(mean_a),
            'mean_b': float(mean_b),
            'std_a': float(std_a),
            'std_b': float(std_b),
            'mean_diff_pct': float(mean_diff_pct),
            'f_ratio': float(f_ratio),
            'means_equivalent': abs(mean_diff_pct) <= tol,
            'tolerance': tol,
        }
    
    return results


def format_campaign_comparison(comparison: Dict[str, Any]) -> str:
    """Format campaign comparison as text."""
    lines = [
        "=" * 60,
        "CAMPAIGN COMPARISON",
        "=" * 60,
        f"Campaign A: {comparison['campaign_a']} (n={comparison['n_tests_a']})",
        f"Campaign B: {comparison['campaign_b']} (n={comparison['n_tests_b']})",
        "",
        f"{'Parameter':<20} {'Mean A':>12} {'Mean B':>12} {'Δ%':>8} {'Status':>8}",
        "-" * 60,
    ]
    
    for param, data in comparison['parameters'].items():
        status = "✓" if data['means_equivalent'] else "✗"
        lines.append(
            f"{param:<20} {data['mean_a']:>12.4g} {data['mean_b']:>12.4g} "
            f"{data['mean_diff_pct']:>+7.2f}% {status:>8}"
        )
    
    return "\n".join(lines)
