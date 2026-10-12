"""Phase 7A: what-if scenario simulator (calculation, validation, data handling, payload)."""
import copy
import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from ai_engine import (
    CONTRACT_VERSION,
    SIMULATION_CONTRACT_VERSION,
    AnalysisConfig,
    RecordValidationError,
    ScenarioValidationError,
    WasteEmissionFactor,
    simulate_payload,
    simulate_scenario,
)

UNITS = {"electricity": "kWh", "water": "liters", "waste": "kg"}
START = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def rec(building, resource, when, value, unit=None):
    ts = when if isinstance(when, str) else when.isoformat()
    return {
        "timestamp": ts, "building": building, "resource_type": resource,
        "consumption_value": value, "unit": unit or UNITS[resource],
    }


def daily(building="BLOCK_A", resource="electricity", days=30, value=100.0, skip=(), start=START):
    """One reading per day for ``days`` days, leaving out the day indexes in ``skip``."""
    return [
        rec(building, resource, start + timedelta(days=i), value)
        for i in range(days) if i not in skip
    ]


def scenario(**kwargs):
    base = {"resource_type": "electricity", "building_code": "BLOCK_A", "reduction_percentage": 15}
    base.update(kwargs)
    return {k: v for k, v in base.items() if v is not None or k == "building_code"}


def all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_keys(v)


def error_locs(exc_info):
    return [tuple(e["loc"]) for e in exc_info.value.errors]


# --------------------------------------------------------------------------- #
# Core arithmetic
# --------------------------------------------------------------------------- #
def test_spec_example_scenario_building_electricity():
    result = simulate_scenario(
        daily(days=45),
        {"resource_type": "electricity", "building_code": "BLOCK_A",
         "reduction_percentage": 15, "baseline_period_days": 30},
    )
    assert result["calculation_status"] == "calculated" and result["not_calculated_reason"] is None
    assert result["baseline"]["value"] == 3000.0
    assert result["estimated_savings"] == {"value": 450.0, "unit": "kWh"}
    assert result["projected_consumption"] == {"value": 2550.0, "unit": "kWh"}
    assert result["reduction_percentage"] == 15.0
    assert result["resource_type"] == "electricity" and result["unit"] == "kWh"
    assert result["scope"] == {"level": "building", "building_code": "BLOCK_A",
                               "buildings_included": ["BLOCK_A"]}
    assert result["data_quality_status"] == "ok"
    assert result["baseline"]["source"] == "measured_records"
    assert result["baseline"]["period_days"] == 30
    assert result["baseline"]["window_start"] == "2026-06-16"
    assert result["baseline"]["window_end"] == "2026-07-15"
    assert result["baseline"]["days_with_data"] == 30 and result["baseline"]["coverage_fraction"] == 1.0
    assert result["baseline"]["readings_used"] == 30


@pytest.mark.parametrize("pct", [0, 0.5, 1, 15, 33.3333, 50, 99.9999, 100])
@pytest.mark.parametrize("value", [0.1, 7.3, 100.0, 12345.678])
def test_savings_projected_identity_and_bounds(pct, value):
    result = simulate_scenario(daily(days=30, value=value), scenario(reduction_percentage=pct))
    baseline = result["baseline"]["value"]
    savings = result["estimated_savings"]["value"]
    projected = result["projected_consumption"]["value"]
    assert baseline == pytest.approx(round(30 * value, 6), abs=1e-6)
    assert savings == pytest.approx(baseline * pct / 100, abs=1e-6)
    assert 0 <= savings <= baseline
    assert projected >= 0
    assert projected == pytest.approx(baseline - savings, abs=1e-6)


def test_zero_and_full_reduction_edges():
    zero = simulate_scenario(daily(), scenario(reduction_percentage=0))
    assert zero["estimated_savings"]["value"] == 0.0 and zero["projected_consumption"]["value"] == 3000.0
    full = simulate_scenario(daily(), scenario(reduction_percentage=100))
    assert full["estimated_savings"]["value"] == 3000.0 and full["projected_consumption"]["value"] == 0.0


def test_percentage_given_as_int_or_float_is_reported_as_float():
    a = simulate_scenario(daily(), scenario(reduction_percentage=15))
    b = simulate_scenario(daily(), scenario(reduction_percentage=15.0))
    assert a["reduction_percentage"] == b["reduction_percentage"] == 15.0
    assert isinstance(a["reduction_percentage"], float)
    assert a["scenario_id"] == b["scenario_id"]


# --------------------------------------------------------------------------- #
# Resources and units
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("resource,unit", [("electricity", "kWh"), ("water", "liters"), ("waste", "kg")])
def test_each_resource_uses_its_standard_unit(resource, unit):
    result = simulate_scenario(
        daily(resource=resource, value=10.0), scenario(resource_type=resource, reduction_percentage=10)
    )
    assert result["unit"] == unit and result["units"]["consumption"] == unit
    assert result["baseline"]["unit"] == unit
    assert result["estimated_savings"]["unit"] == unit and result["projected_consumption"]["unit"] == unit
    assert result["baseline"]["value"] == 300.0 and result["estimated_savings"]["value"] == 30.0


def test_unit_spelling_aliases_are_normalized_by_existing_schema():
    records = [
        rec("Tank", "water", START + timedelta(days=i), 50.0, unit=("litres" if i % 2 else "L"))
        for i in range(30)
    ]
    result = simulate_scenario(records, scenario(resource_type="water", building_code="Tank"))
    assert result["unit"] == "liters" and result["baseline"]["value"] == 1500.0


@pytest.mark.parametrize("resource,unit", [("electricity", "MWh"), ("electricity", "Wh"),
                                           ("water", "gallons"), ("waste", "lbs")])
def test_incompatible_record_units_are_rejected_not_converted(resource, unit):
    records = daily(resource=resource)
    records[3] = rec("BLOCK_A", resource, START + timedelta(days=3), 1.0, unit=unit)
    with pytest.raises(RecordValidationError) as exc_info:
        simulate_scenario(records, scenario(resource_type=resource))
    assert exc_info.value.errors[0]["index"] == 3


def test_other_resources_are_never_mixed_into_the_baseline():
    records = daily(resource="electricity", value=100.0) + daily(resource="water", value=9999.0) \
        + daily(resource="waste", value=7.0)
    result = simulate_scenario(records, scenario(resource_type="electricity"))
    assert result["baseline"]["value"] == 3000.0 and result["baseline"]["readings_used"] == 30
    water = simulate_scenario(records, scenario(resource_type="water"))
    assert water["baseline"]["value"] == 9999.0 * 30 and water["unit"] == "liters"


def test_resource_type_is_case_and_whitespace_insensitive():
    assert simulate_scenario(daily(), scenario(resource_type="  Electricity "))["resource_type"] == "electricity"


# --------------------------------------------------------------------------- #
# Scope: building vs campus
# --------------------------------------------------------------------------- #
def test_campus_wide_sums_all_buildings_and_lists_breakdown():
    records = daily("BLOCK_A", value=100.0) + daily("LIBRARY", value=50.0) + daily("HOSTEL", value=25.0)
    result = simulate_scenario(records, scenario(building_code=None, reduction_percentage=20))
    assert result["scope"]["level"] == "campus" and result["scope"]["building_code"] is None
    assert result["scope"]["buildings_included"] == ["BLOCK_A", "HOSTEL", "LIBRARY"]
    assert result["baseline"]["value"] == 30 * 175.0
    assert result["estimated_savings"]["value"] == 30 * 175.0 * 0.2
    assert [row["building"] for row in result["building_breakdown"]] == ["BLOCK_A", "HOSTEL", "LIBRARY"]
    assert sum(row["baseline_value"] for row in result["building_breakdown"]) == result["baseline"]["value"]
    assert result["data_quality_status"] == "ok"


def test_campus_scope_omitted_key_is_same_as_null():
    records = daily("A") + daily("B")
    without_key = {"resource_type": "electricity", "reduction_percentage": 10}
    assert simulate_scenario(records, without_key) == simulate_scenario(records, {**without_key, "building_code": None})


def test_building_scope_ignores_other_buildings():
    records = daily("BLOCK_A", value=100.0) + daily("LIBRARY", value=50.0)
    result = simulate_scenario(records, scenario(building_code="LIBRARY"))
    assert result["baseline"]["value"] == 1500.0 and result["building_breakdown"] is None


def test_building_match_is_case_insensitive_and_reports_the_data_spelling():
    result = simulate_scenario(daily("Block_A"), scenario(building_code="  block_a "))
    assert result["calculation_status"] == "calculated"
    assert result["scope"]["building_code"] == "Block_A"


def test_building_names_differing_only_in_case_are_flagged_suspect():
    records = daily("BLOCK_A", days=15) + daily("block_a", days=15, start=START + timedelta(days=15))
    result = simulate_scenario(records, scenario(building_code="Block_A"))
    assert result["baseline"]["value"] == 3000.0
    assert result["data_quality_status"] == "suspect"
    assert any("letter case" in w for w in result["warnings"])


# --------------------------------------------------------------------------- #
# Data conditions that give a normal "not calculated" result
# --------------------------------------------------------------------------- #
def assert_not_calculated(result, reason):
    assert result["calculation_status"] == "not_calculated"
    assert result["not_calculated_reason"] == reason
    assert result["baseline"]["value"] is None
    assert result["estimated_savings"]["value"] is None and result["projected_consumption"]["value"] is None
    assert result["estimated_co2e_savings_kg"] is None
    assert result["data_quality_status"] == "insufficient_data"
    assert result["warnings"]
    assert result["projection_type"] == "hypothetical" and result["measured_savings"] is None


def test_empty_dataset():
    assert_not_calculated(simulate_scenario([], scenario()), "no_records")
    assert_not_calculated(simulate_scenario(iter([]), scenario(building_code=None)), "no_records")


def test_unknown_building_lists_known_buildings():
    result = simulate_scenario(daily("BLOCK_A") + daily("LIBRARY"), scenario(building_code="BLOCK_Z"))
    assert_not_calculated(result, "unknown_building")
    assert "BLOCK_Z" in result["warnings"][0] and "'LIBRARY'" in result["warnings"][0]
    assert result["scope"]["buildings_included"] == []


def test_building_without_records_for_the_requested_resource():
    result = simulate_scenario(daily("BLOCK_A", resource="electricity"), scenario(resource_type="water"))
    assert_not_calculated(result, "no_resource_records")


def test_campus_without_records_for_the_requested_resource():
    result = simulate_scenario(daily("BLOCK_A", resource="electricity"),
                               scenario(resource_type="waste", building_code=None))
    assert_not_calculated(result, "no_resource_records")


def test_baseline_period_without_enough_data_is_not_extrapolated():
    skip = set(range(0, 30, 3))  # 10 of 30 days missing -> 20/30 = 67 % < 80 %
    result = simulate_scenario(daily(skip=skip), scenario())
    assert_not_calculated(result, "insufficient_coverage")
    assert result["baseline"]["days_with_data"] == 20 and result["baseline"]["period_days"] == 30
    assert any("20 of the 30 days" in w for w in result["warnings"])


def test_only_one_day_of_data_does_not_become_a_30_day_baseline():
    assert_not_calculated(simulate_scenario(daily(days=1), scenario()), "insufficient_coverage")
    one_day = simulate_scenario(daily(days=1), scenario(baseline_period_days=1))
    assert one_day["calculation_status"] == "calculated" and one_day["baseline"]["value"] == 100.0


# --------------------------------------------------------------------------- #
# Baseline window, partial coverage, duplicates, irregular data
# --------------------------------------------------------------------------- #
def test_only_the_most_recent_period_is_used_and_it_ends_at_the_last_reading():
    values = [1000.0] * 30 + [100.0] * 30  # old days high, recent days low
    records = [rec("BLOCK_A", "electricity", START + timedelta(days=i), v) for i, v in enumerate(values)]
    result = simulate_scenario(records, scenario(baseline_period_days=30))
    assert result["baseline"]["value"] == 3000.0
    assert result["baseline"]["window_end"] == "2026-07-30"
    assert any("older than the baseline window" in a for a in result["assumptions"])


def test_default_baseline_period_is_30_days_and_custom_period_is_respected():
    assert simulate_scenario(daily(days=40), scenario())["baseline"]["period_days"] == 30
    seven = simulate_scenario(daily(days=40), scenario(baseline_period_days=7))
    assert seven["baseline"]["period_days"] == 7 and seven["baseline"]["value"] == 700.0
    assert any("7 days" in a for a in seven["assumptions"])


def test_partial_coverage_is_flagged_and_baseline_is_not_gap_filled():
    result = simulate_scenario(daily(skip={3, 7, 11, 15, 19}), scenario())  # 25 of 30 days
    assert result["calculation_status"] == "calculated"
    assert result["baseline"]["value"] == 2500.0  # not scaled up to 3000
    assert result["baseline"]["days_with_data"] == 25 and result["baseline"]["coverage_fraction"] == 0.833333
    assert result["data_quality_status"] == "suspect"
    assert any("25 of the 30 days" in w and "understates" in w for w in result["warnings"])


def test_campus_building_with_partial_window_is_flagged():
    records = daily("A", days=30) + daily("B", days=30, skip=set(range(10)))
    result = simulate_scenario(records, scenario(building_code=None))
    assert result["data_quality_status"] == "suspect"
    assert any("'B'" in w and "partial" in w for w in result["warnings"])
    rows = {r["building"]: r for r in result["building_breakdown"]}
    assert rows["A"]["days_with_data"] == 30 and rows["B"]["days_with_data"] == 20


def test_campus_building_that_stopped_reporting_is_flagged():
    records = daily("A", days=60) + daily("OLD", days=10)
    result = simulate_scenario(records, scenario(building_code=None))
    assert result["data_quality_status"] == "suspect"
    assert any("no readings inside the baseline window" in w and "'OLD'" in w for w in result["warnings"])
    assert result["scope"]["buildings_included"] == ["A"]


def test_exact_duplicate_readings_count_once():
    records = daily() + [daily()[5]] * 3
    result = simulate_scenario(records, scenario())
    assert result["baseline"]["value"] == 3000.0 and result["baseline"]["readings_used"] == 30
    assert any("3 exact duplicate" in w for w in result["warnings"])
    assert result["data_quality_status"] == "ok"


def test_conflicting_duplicates_are_excluded_and_flagged():
    records = daily() + [rec("BLOCK_A", "electricity", START + timedelta(days=5), 150.0)]
    result = simulate_scenario(records, scenario())
    assert result["baseline"]["value"] == 2900.0 and result["baseline"]["days_with_data"] == 29
    assert result["data_quality_status"] == "suspect"
    assert any("conflicting" in w for w in result["warnings"])


def test_all_readings_conflicting_gives_not_calculated():
    records = [rec("A", "electricity", START, 1.0), rec("A", "electricity", START, 2.0)]
    assert_not_calculated(simulate_scenario(records, scenario(building_code="A", baseline_period_days=1)),
                          "no_usable_readings")


def test_irregular_reading_frequency_is_flagged():
    records = daily() + [rec("BLOCK_A", "electricity", START + timedelta(days=10, hours=3), 40.0)]
    result = simulate_scenario(records, scenario())
    assert result["baseline"]["value"] == 3040.0 and result["baseline"]["readings_used"] == 31
    assert result["data_quality_status"] == "suspect"
    assert any("irregular" in w for w in result["warnings"])


def test_regular_sub_daily_readings_are_summed_without_warning():
    records = [
        rec("BLOCK_A", "electricity", START.replace(hour=0) + timedelta(days=d, hours=h), 4.0)
        for d in range(30) for h in range(0, 24, 6)
    ]
    result = simulate_scenario(records, scenario())
    assert result["baseline"]["value"] == 30 * 4 * 4.0 and result["data_quality_status"] == "ok"


def test_mixed_naive_and_aware_timestamps_are_flagged():
    records = daily()
    for i in range(15):
        records[i] = rec("BLOCK_A", "electricity", (START + timedelta(days=i)).replace(tzinfo=None).isoformat(), 100.0)
    result = simulate_scenario(records, scenario())
    assert result["baseline"]["value"] == 3000.0 and result["data_quality_status"] == "suspect"
    assert any("15 timestamp(s) had no UTC offset" in w for w in result["warnings"])


def test_calendar_day_is_the_utc_day():
    records = [rec("A", "electricity", "2026-06-01T23:30:00-05:00", 40.0)]  # = 2026-06-02 04:30 UTC
    result = simulate_scenario(records, scenario(building_code="A", baseline_period_days=1))
    assert result["baseline"]["window_end"] == "2026-06-02" and result["baseline"]["value"] == 40.0


def test_result_does_not_depend_on_input_order():
    records = daily(days=45) + daily("LIBRARY", days=45, value=20.0)
    forward = simulate_scenario(records, scenario(building_code=None))
    backward = simulate_scenario(list(reversed(records)), scenario(building_code=None))
    assert forward == backward


def test_inputs_are_not_mutated():
    records, spec = daily(days=40), scenario(baseline_period_days=30)
    before = (copy.deepcopy(records), copy.deepcopy(spec))
    simulate_scenario(records, spec)
    assert (records, spec) == before


# --------------------------------------------------------------------------- #
# Zero baseline
# --------------------------------------------------------------------------- #
def test_zero_baseline_gives_zero_savings_and_a_suspect_flag():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.5,
                         electricity_emission_factor_source="test fixture")
    result = simulate_scenario(daily(value=0.0), scenario(), cfg)
    assert result["calculation_status"] == "calculated"
    assert result["baseline"]["value"] == 0.0
    assert result["estimated_savings"]["value"] == 0.0 and result["projected_consumption"]["value"] == 0.0
    assert result["estimated_co2e_savings_kg"] == 0.0
    assert result["data_quality_status"] == "suspect"
    assert any("baseline is 0" in w for w in result["warnings"])


def test_explicit_zero_baseline():
    result = simulate_scenario([], scenario(baseline_consumption={"value": 0, "unit": "kWh", "period_days": 30}))
    assert result["estimated_savings"]["value"] == 0.0 and result["data_quality_status"] == "suspect"


# --------------------------------------------------------------------------- #
# Explicit baseline
# --------------------------------------------------------------------------- #
def test_explicit_baseline_needs_no_records():
    result = simulate_scenario(
        [], scenario(baseline_consumption={"value": 4200, "unit": "kWh", "period_days": 30})
    )
    assert result["calculation_status"] == "calculated"
    assert result["baseline"]["value"] == 4200.0 and result["baseline"]["source"] == "explicit"
    assert result["baseline"]["period_days"] == 30
    assert result["baseline"]["window_start"] is None and result["baseline"]["days_with_data"] is None
    assert result["estimated_savings"]["value"] == 630.0 and result["projected_consumption"]["value"] == 3570.0
    assert result["data_quality_status"] == "ok"
    assert any("not verified" in a for a in result["assumptions"])


def test_explicit_baseline_wins_over_records_and_says_so():
    result = simulate_scenario(
        daily(value=1.0), scenario(baseline_consumption={"value": 4200, "unit": "kwh", "period_days": 30})
    )
    assert result["baseline"]["value"] == 4200.0
    assert any("not used" in w for w in result["warnings"])


def test_explicit_baseline_in_other_units_per_resource():
    water = simulate_scenario([], scenario(resource_type="water", reduction_percentage=10,
                                           baseline_consumption={"value": 250000, "unit": "litres", "period_days": 7}))
    assert water["unit"] == "liters" and water["estimated_savings"] == {"value": 25000.0, "unit": "liters"}
    assert water["baseline"]["period_days"] == 7
    waste = simulate_scenario([], scenario(resource_type="waste", building_code=None, reduction_percentage=25,
                                           baseline_consumption={"value": 800, "unit": "KG", "period_days": 30}))
    assert waste["estimated_savings"]["value"] == 200.0 and waste["scope"]["level"] == "campus"


def test_explicit_baseline_period_may_be_given_once_or_twice_if_equal():
    explicit = {"value": 100, "unit": "kWh", "period_days": 14}
    once = simulate_scenario([], scenario(baseline_consumption=explicit))
    twice = simulate_scenario([], scenario(baseline_consumption=explicit, baseline_period_days=14))
    assert once["baseline"]["period_days"] == twice["baseline"]["period_days"] == 14


@pytest.mark.parametrize("explicit,loc", [
    ({"value": 100, "unit": "MWh", "period_days": 30}, ("baseline_consumption", "unit")),
    ({"value": 100, "unit": "liters", "period_days": 30}, ("baseline_consumption", "unit")),
    ({"value": 100, "unit": 5, "period_days": 30}, ("baseline_consumption", "unit")),
    ({"value": -1, "unit": "kWh", "period_days": 30}, ("baseline_consumption", "value")),
    ({"value": "100", "unit": "kWh", "period_days": 30}, ("baseline_consumption", "value")),
    ({"value": float("nan"), "unit": "kWh", "period_days": 30}, ("baseline_consumption", "value")),
    ({"value": True, "unit": "kWh", "period_days": 30}, ("baseline_consumption", "value")),
    ({"value": 100, "unit": "kWh", "period_days": 0}, ("baseline_consumption", "period_days")),
    ({"value": 100, "unit": "kWh", "period_days": 2.5}, ("baseline_consumption", "period_days")),
    ({"value": 100, "unit": "kWh"}, ("baseline_consumption", "period_days")),
    ({"unit": "kWh", "period_days": 30}, ("baseline_consumption", "value")),
    ({"value": 100, "period_days": 30}, ("baseline_consumption", "unit")),
    ({"value": 100, "unit": "kWh", "period_days": 30, "currency": "INR"}, ("baseline_consumption", "currency")),
    ("4200 kWh", ("baseline_consumption",)),
])
def test_invalid_explicit_baselines_are_rejected(explicit, loc):
    with pytest.raises(ScenarioValidationError) as exc_info:
        simulate_scenario([], scenario(baseline_consumption=explicit))
    assert loc in error_locs(exc_info)


def test_explicit_baseline_period_conflict_is_rejected_not_rescaled():
    with pytest.raises(ScenarioValidationError) as exc_info:
        simulate_scenario([], scenario(baseline_period_days=30,
                                       baseline_consumption={"value": 100, "unit": "kWh", "period_days": 7}))
    assert ("baseline_period_days",) in error_locs(exc_info)
    assert "never rescaled" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Scenario validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pct", [-0.0001, -1, -100, 100.0001, 101, 1000, float("nan"), float("inf"),
                                 float("-inf"), "15", "15%", True, False, [15], {"v": 15}])
def test_invalid_reduction_percentages_are_rejected(pct):
    with pytest.raises(ScenarioValidationError) as exc_info:
        simulate_scenario(daily(), scenario(reduction_percentage=pct))
    assert ("reduction_percentage",) in error_locs(exc_info)


def test_missing_reduction_percentage_is_rejected():
    with pytest.raises(ScenarioValidationError) as exc_info:
        simulate_scenario(daily(), {"resource_type": "electricity", "building_code": "BLOCK_A"})
    assert error_locs(exc_info) == [("reduction_percentage",)]


def test_out_of_range_message_names_the_range():
    with pytest.raises(ScenarioValidationError) as exc_info:
        simulate_scenario(daily(), scenario(reduction_percentage=120))
    assert "between 0 and 100" in str(exc_info.value)


@pytest.mark.parametrize("resource", ["gas", "", "ELECTRIC", 5, ["water"], None])
def test_unsupported_or_missing_resource_type_is_rejected(resource):
    spec = {"resource_type": resource, "reduction_percentage": 10}
    with pytest.raises(ScenarioValidationError) as exc_info:
        simulate_scenario(daily(), spec)
    assert ("resource_type",) in error_locs(exc_info)


@pytest.mark.parametrize("days", [0, -5, 367, 2.5, "30", True, float("nan"), float("inf"), [30]])
def test_invalid_baseline_period_days_are_rejected(days):
    with pytest.raises(ScenarioValidationError) as exc_info:
        simulate_scenario(daily(), scenario(baseline_period_days=days))
    assert ("baseline_period_days",) in error_locs(exc_info)


def test_integral_float_period_is_accepted():
    assert simulate_scenario(daily(), scenario(baseline_period_days=30.0))["baseline"]["period_days"] == 30


@pytest.mark.parametrize("building", ["", "   ", 5, ["A"], True])
def test_invalid_building_code_is_rejected(building):
    with pytest.raises(ScenarioValidationError) as exc_info:
        simulate_scenario(daily(), scenario(building_code=building))
    assert ("building_code",) in error_locs(exc_info)


def test_unknown_scenario_keys_are_rejected():
    with pytest.raises(ScenarioValidationError) as exc_info:
        simulate_scenario(daily(), scenario(cost_per_kwh=8))
    assert ("cost_per_kwh",) in error_locs(exc_info)


def test_all_scenario_problems_are_reported_together():
    with pytest.raises(ScenarioValidationError) as exc_info:
        simulate_scenario([], {"resource_type": "gas", "reduction_percentage": 150,
                               "baseline_period_days": 0, "extra": 1})
    locs = error_locs(exc_info)
    assert {("resource_type",), ("reduction_percentage",), ("baseline_period_days",), ("extra",)} <= set(locs)


def test_scenario_is_validated_before_records():
    with pytest.raises(ScenarioValidationError):
        simulate_scenario([{"not": "a record"}], scenario(reduction_percentage=-1))


def test_scenario_validation_error_is_a_value_error():
    assert issubclass(ScenarioValidationError, ValueError)


def test_wrong_argument_types_raise_type_error():
    with pytest.raises(TypeError):
        simulate_scenario(daily(), [("resource_type", "water")])
    with pytest.raises(TypeError):
        simulate_scenario(None, scenario())
    with pytest.raises(TypeError):
        simulate_scenario("records", scenario())
    with pytest.raises(TypeError):
        simulate_scenario(daily(), scenario(), config={"electricity_emission_factor_kg_co2e_per_kwh": 1})


def test_invalid_records_raise_the_existing_record_validation_error():
    bad = daily()
    bad[2] = {"timestamp": "nope", "building": "A", "resource_type": "electricity",
              "consumption_value": -4, "unit": "kWh"}
    with pytest.raises(RecordValidationError) as exc_info:
        simulate_scenario(bad, scenario())
    assert exc_info.value.errors[0]["index"] == 2


# --------------------------------------------------------------------------- #
# CO2e
# --------------------------------------------------------------------------- #
def test_electricity_co2e_uses_the_configured_factor_and_source():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.5,
                         electricity_emission_factor_source="test fixture")
    result = simulate_scenario(daily(), scenario(), cfg)
    assert result["estimated_co2e_savings_kg"] == pytest.approx(225.0)
    assert result["emission_factor"] == {"value": 0.5, "unit": "kg CO2e/kWh"}
    assert result["units"]["co2e"] == "kg CO2e"
    assert any("test fixture" in a for a in result["assumptions"])
    assert not any("CO2e" in w for w in result["warnings"])


def test_missing_emission_factor_means_no_co2e_and_a_warning():
    for resource in ("electricity", "water", "waste"):
        result = simulate_scenario(daily(resource=resource), scenario(resource_type=resource))
        assert result["calculation_status"] == "calculated"
        assert result["estimated_co2e_savings_kg"] is None and result["emission_factor"] is None
        assert result["estimated_savings"]["value"] > 0  # physical savings are still reported
        assert any("CO2e" in w and "not calculated" in w or "no carbon-equivalence" in w
                   for w in result["warnings"])


def test_water_and_waste_factors_follow_the_existing_config_conventions():
    cfg = AnalysisConfig(
        water_emission_factor_kg_co2e_per_liter=0.001,
        water_emission_factor_source="test fixture",
        waste_emission_factor=WasteEmissionFactor(value=2.0, unit="kg_co2e_per_kg", source="test fixture"),
    )
    water = simulate_scenario(daily(resource="water", value=100.0),
                              scenario(resource_type="water", reduction_percentage=10), cfg)
    assert water["estimated_co2e_savings_kg"] == pytest.approx(300.0 * 0.001)
    waste = simulate_scenario(daily(resource="waste", value=10.0),
                              scenario(resource_type="waste", reduction_percentage=10), cfg)
    assert waste["estimated_co2e_savings_kg"] == pytest.approx(30.0 * 2.0)
    assert waste["emission_factor"]["unit"] == "kg CO2e/kg"


def test_factor_for_one_resource_is_not_applied_to_another():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.5)
    water = simulate_scenario(daily(resource="water"), scenario(resource_type="water"), cfg)
    assert water["estimated_co2e_savings_kg"] is None


def test_zero_factor_is_a_valid_factor_and_gives_zero_co2e():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.0)
    assert simulate_scenario(daily(), scenario(), cfg)["estimated_co2e_savings_kg"] == 0.0


def test_not_calculated_result_has_no_co2e_even_with_a_factor():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.5)
    result = simulate_scenario([], scenario(), cfg)
    assert result["estimated_co2e_savings_kg"] is None and result["emission_factor"] is None


# --------------------------------------------------------------------------- #
# Hypothetical vs measured, no money
# --------------------------------------------------------------------------- #
def test_result_is_labelled_hypothetical_never_measured():
    for result in (
        simulate_scenario(daily(), scenario()),
        simulate_scenario([], scenario()),
        simulate_scenario([], scenario(baseline_consumption={"value": 1, "unit": "kWh", "period_days": 30})),
    ):
        assert result["projection_type"] == "hypothetical" and result["measured_savings"] is None
    result = simulate_scenario(daily(), scenario())
    assert result["assumptions"][0].startswith("Hypothetical projection")
    assert "not measured savings" in result["assumptions"][0]


def test_no_monetary_fields_or_claims_anywhere():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.5)
    forbidden = ("cost", "price", "money", "monetary", "currency", "usd", "inr", "rupee", "dollar", "tariff", "bill")
    for result in (simulate_scenario(daily(), scenario(), cfg), simulate_scenario([], scenario())):
        keys = [str(k).lower() for k in all_keys(result)]
        assert not [k for k in keys if any(f in k for f in forbidden)]
    assert "No monetary savings are estimated." in simulate_scenario(daily(), scenario())["assumptions"]


def test_assumptions_state_formulas_units_and_period():
    result = simulate_scenario(daily(), scenario())
    text = " ".join(result["assumptions"])
    assert "baseline x reduction_percentage / 100" in text and "kWh" in text and "30 days" in text
    assert "never rescaled" in text


# --------------------------------------------------------------------------- #
# Scenario ID, JSON safety
# --------------------------------------------------------------------------- #
def test_scenario_id_is_deterministic_and_depends_on_the_definition():
    base = scenario()
    a = simulate_scenario(daily(), base)["scenario_id"]
    assert a.startswith("scn-") and len(a) == 16
    assert simulate_scenario(daily(days=50, value=3.0), base)["scenario_id"] == a  # data does not matter
    assert simulate_scenario(daily(), {**base, "building_code": "block_a"})["scenario_id"] == a
    variants = [
        {**base, "reduction_percentage": 16},
        {**base, "baseline_period_days": 7},
        {**base, "building_code": None},
        {**base, "resource_type": "water"},
        {**base, "baseline_consumption": {"value": 5, "unit": "kWh", "period_days": 30}},
    ]
    ids = {simulate_scenario(daily(), v)["scenario_id"] for v in variants}
    assert a not in ids and len(ids) == len(variants)


def test_custom_scenario_id_is_used_and_validated():
    assert simulate_scenario(daily(), scenario(scenario_id="pilot-2026.A"))["scenario_id"] == "pilot-2026.A"
    for bad in ("", " ", "has space", "x" * 65, "-leading", 7):
        with pytest.raises(ScenarioValidationError):
            simulate_scenario(daily(), scenario(scenario_id=bad))


def test_result_is_strictly_json_safe_and_round_trips():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.4)
    for result in (
        simulate_scenario(daily(skip={2, 4}), scenario(building_code=None), cfg),
        simulate_scenario([], scenario()),
    ):
        text = json.dumps(result, allow_nan=False)
        assert json.loads(text) == result


def test_documented_top_level_keys_are_all_present():
    result = simulate_scenario(daily(), scenario())
    assert set(result) == {
        "simulation_contract_version", "engine_version", "scenario_id", "projection_type",
        "calculation_status", "not_calculated_reason", "scope", "resource_type", "unit", "baseline",
        "reduction_percentage", "projected_consumption", "estimated_savings", "estimated_co2e_savings_kg",
        "emission_factor", "measured_savings", "units", "building_breakdown", "assumptions", "warnings",
        "data_quality_status",
    }
    assert set(result["baseline"]) == {
        "value", "unit", "period_days", "source", "window_start", "window_end",
        "days_with_data", "coverage_fraction", "readings_used",
    }
    assert result["simulation_contract_version"] == SIMULATION_CONTRACT_VERSION
    assert result["data_quality_status"] in {"ok", "suspect", "insufficient_data"}


# --------------------------------------------------------------------------- #
# simulate_payload (JSON entry point)
# --------------------------------------------------------------------------- #
def test_payload_success_envelope_wraps_the_same_result():
    request = {"records": daily(days=45), "scenario": scenario(baseline_period_days=30)}
    outcome = simulate_payload(request)
    assert outcome.ok and outcome.http_status == 200
    body = outcome.body
    assert body["status"] == "ok" and body["contract_version"] == CONTRACT_VERSION
    assert body["simulation_contract_version"] == SIMULATION_CONTRACT_VERSION
    assert body["result"] == simulate_scenario(request["records"], request["scenario"])
    json.dumps(body, allow_nan=False)


def test_payload_uses_the_supplied_config():
    cfg = AnalysisConfig(electricity_emission_factor_kg_co2e_per_kwh=0.5)
    body = simulate_payload({"records": daily(), "scenario": scenario()}, cfg).body
    assert body["result"]["estimated_co2e_savings_kg"] == pytest.approx(225.0)


def test_payload_records_are_optional_with_an_explicit_baseline():
    body = simulate_payload({"scenario": scenario(
        baseline_consumption={"value": 4200, "unit": "kWh", "period_days": 30})}).body
    assert body["status"] == "ok" and body["result"]["estimated_savings"]["value"] == 630.0
    assert simulate_payload({"records": None, "scenario": scenario(
        baseline_consumption={"value": 4200, "unit": "kWh", "period_days": 30})}).http_status == 200


def test_payload_not_calculated_is_http_200():
    outcome = simulate_payload({"records": daily("A"), "scenario": scenario(building_code="B")})
    assert outcome.http_status == 200 and outcome.body["result"]["calculation_status"] == "not_calculated"
    assert simulate_payload({"scenario": scenario()}).body["result"]["not_calculated_reason"] == "no_records"


def test_payload_invalid_scenario_is_a_contract_error_with_field_paths():
    outcome = simulate_payload({"records": daily(), "scenario": scenario(reduction_percentage=101)})
    assert not outcome.ok and outcome.http_status == 422
    error = outcome.body["error"]
    assert outcome.body["status"] == "error" and error["code"] == "invalid_scenario"
    assert error["details"][0]["errors"][0]["loc"] == ["scenario", "reduction_percentage"]


def test_payload_invalid_records_is_a_contract_error():
    bad = daily()
    bad[1] = {"timestamp": "x"}
    outcome = simulate_payload({"records": bad, "scenario": scenario()})
    assert outcome.http_status == 422 and outcome.body["error"]["code"] == "invalid_records"
    assert outcome.body["error"]["details"][0]["index"] == 1


@pytest.mark.parametrize("payload", [[], "x", None, 5, {"scenario": scenario(), "extra": 1}, {"records": []},
                                     {"scenario": "x"}, {"scenario": [1]},
                                     {"records": "nope", "scenario": scenario()},
                                     {"records": {"a": 1}, "scenario": scenario()}])
def test_payload_bad_shapes_are_invalid_request(payload):
    outcome = simulate_payload(payload)
    assert outcome.http_status == 422 and outcome.body["error"]["code"] == "invalid_request"


def test_payload_record_limit():
    outcome = simulate_payload({"records": daily(), "scenario": scenario()}, max_records=3)
    assert outcome.http_status == 413 and outcome.body["error"]["code"] == "too_many_records"


def test_payload_wrong_config_type_is_a_programmer_error():
    with pytest.raises(TypeError):
        simulate_payload({"records": daily(), "scenario": scenario()}, config="nope")
