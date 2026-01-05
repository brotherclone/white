from app.structures.enums.creative_transformation_type import CreativeTransformationType


def test_enum_values():
    assert (
        CreativeTransformationType.PHYSICAL.value == "physical_trauma_aestheticization"
    )
    assert CreativeTransformationType.SPATIAL.value == "spatial_uncertainty_exploration"
    assert (
        CreativeTransformationType.PERCEPTUAL.value
        == "perceptual_instability_investigation"
    )
    assert CreativeTransformationType.TEMPORAL.value == "temporal_boundary_dissolution"
    assert CreativeTransformationType.IDENTITY.value == "identity_fluidity_exploration"
    assert CreativeTransformationType.GENERAL.value == "general_rebracketing_pattern"


def test_enum_membership():
    assert "PHYSICAL" in CreativeTransformationType.__members__
    assert "SPATIAL" in CreativeTransformationType.__members__
    assert "PERCEPTUAL" in CreativeTransformationType.__members__
    assert "TEMPORAL" in CreativeTransformationType.__members__
    assert "IDENTITY" in CreativeTransformationType.__members__
    assert "GENERAL" in CreativeTransformationType.__members__


def test_enum_count():
    assert len(CreativeTransformationType) == 6


def test_enum_iteration():
    types = list(CreativeTransformationType)
    assert len(types) == 6
    assert CreativeTransformationType.PHYSICAL in types


def test_string_comparison():
    assert CreativeTransformationType.PHYSICAL == "physical_trauma_aestheticization"
    assert CreativeTransformationType.TEMPORAL == "temporal_boundary_dissolution"
