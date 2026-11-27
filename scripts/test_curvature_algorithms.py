import numpy as np

from test_terrain_features_v2 import (
    calculate_features,
    run_algorithm_tests,
    NODATA
)


print("=" * 65)
print("FINAL TERRAIN ALGORITHM VALIDATION")
print("=" * 65)

run_algorithm_tests()

RES = 30.0

coordinates = np.arange(-2, 3) * RES

x, y = np.meshgrid(coordinates, coordinates)

valid = np.ones((5, 5), dtype=bool)

gradient = 0.5
curvature = 0.002


def centre(result, feature):
    return result[feature][1, 1]


# ============================================================
# TEST 1: CONVEX PROFILE
# ============================================================

dem = (
    100
    + gradient * x
    - 0.5 * curvature * x**2
)

result = calculate_features(dem, valid, RES, RES)

expected = curvature / (1 + gradient**2)**1.5

np.testing.assert_allclose(
    centre(result, "profile_curvature"),
    expected,
    atol=1e-10
)

np.testing.assert_allclose(
    centre(result, "plan_curvature"),
    0,
    atol=1e-10
)

print("\nConvex profile test: PASS")


# ============================================================
# TEST 2: CONCAVE PROFILE
# ============================================================

dem = (
    100
    + gradient * x
    + 0.5 * curvature * x**2
)

result = calculate_features(dem, valid, RES, RES)

np.testing.assert_allclose(
    centre(result, "profile_curvature"),
    -expected,
    atol=1e-10
)

print("Concave profile test: PASS")


# ============================================================
# TEST 3: CONVEX CROSS-SLOPE
# ============================================================

dem = (
    100
    + gradient * x
    - 0.5 * curvature * y**2
)

result = calculate_features(dem, valid, RES, RES)

plan = centre(result, "plan_curvature")
slope = centre(result, "slope")

tangential = plan * np.sin(np.radians(slope))

expected_plan = curvature / gradient

expected_tangential = (
    curvature / np.sqrt(1 + gradient**2)
)

np.testing.assert_allclose(
    plan, expected_plan, atol=1e-10
)

np.testing.assert_allclose(
    tangential, expected_tangential, atol=1e-10
)

print("Convex cross-slope test: PASS")


# ============================================================
# TEST 4: CONCAVE CROSS-SLOPE
# ============================================================

dem = (
    100
    + gradient * x
    + 0.5 * curvature * y**2
)

result = calculate_features(dem, valid, RES, RES)

plan = centre(result, "plan_curvature")
slope = centre(result, "slope")

tangential = plan * np.sin(np.radians(slope))

np.testing.assert_allclose(
    plan, -expected_plan, atol=1e-10
)

np.testing.assert_allclose(
    tangential, -expected_tangential, atol=1e-10
)

print("Concave cross-slope test: PASS")


# ============================================================
# TEST 5: MIXED DERIVATIVE AND DIAGONAL SLOPE
# ============================================================

p = 0.4
q = 0.3

r = 0.002
s = 0.001
t = -0.003

dem = (
    100
    + p * x
    + q * y
    + 0.5 * r * x**2
    + s * x * y
    + 0.5 * t * y**2
)

result = calculate_features(dem, valid, RES, RES)

g = np.hypot(p, q)

hessian = np.array([
    [r, s],
    [s, t]
])

along = np.array([p, q]) / g
across = np.array([-q, p]) / g

expected_profile = (
    -(along @ hessian @ along)
    / (1 + g**2)**1.5
)

expected_plan = (
    -(across @ hessian @ across) / g
)

np.testing.assert_allclose(
    centre(result, "profile_curvature"),
    expected_profile,
    atol=1e-10
)

np.testing.assert_allclose(
    centre(result, "plan_curvature"),
    expected_plan,
    atol=1e-10
)

print("Mixed derivative test: PASS")


# ============================================================
# TEST 6: NEARLY FLAT TERRAIN
# ============================================================

dem = 100 + 0.000001 * x

result = calculate_features(dem, valid, RES, RES)

assert centre(result, "plan_curvature") == NODATA
assert centre(result, "profile_curvature") == NODATA

print("Near-flat terrain test: PASS")


print("\n" + "=" * 65)
print("ALL TERRAIN ALGORITHM TESTS PASSED")
print("=" * 65)