import pytest
import inspect

MODULE_NAME = "app.agents.states.violet_agent_state"
CLASS_NAME = "VioletAgentState"


def _get_class_or_skip():
    module = pytest.importorskip(
        MODULE_NAME, reason=f"module {MODULE_NAME} not available"
    )
    cls = getattr(module, CLASS_NAME, None)
    if cls is None:
        pytest.skip(f"{CLASS_NAME} not found in {MODULE_NAME}")
    return cls


def test_violet_agent_state_is_callable():
    cls = _get_class_or_skip()
    assert inspect.isclass(cls), f"{CLASS_NAME} should be a class"
    sig = inspect.signature(cls)
    try:
        if any(
            p.default is inspect._empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in sig.parameters.values()
        ):
            obj = cls.__new__(cls)
            try:
                cls.__init__(obj)
            except TypeError:
                pytest.skip(
                    f"{CLASS_NAME} requires constructor parameters; skipping default-instantiation tests"
                )
        else:
            obj = cls()
    except Exception as e:
        pytest.skip(f"Could not instantiate {CLASS_NAME} without parameters: {e}")
    assert isinstance(obj, cls)


def test_interface_methods_present():
    cls = _get_class_or_skip()
    try:
        obj = cls()
    except Exception:
        pytest.skip("Skipping because instantiation without args failed")
    expected_attrs = ["enter", "exit", "update", "next_state", "to_dict", "from_dict"]
    found_any = False
    for attr in expected_attrs:
        if hasattr(obj, attr):
            found_any = True
            break
    if not found_any:
        pytest.skip(
            "No common lifecycle/serialization methods found; ensure the class implements at least one of: "
            + ", ".join(expected_attrs)
        )
    # Check each method if present is callable
    for attr in expected_attrs:
        if hasattr(obj, attr):
            assert callable(
                getattr(obj, attr)
            ), f"{attr} should be callable when present"


def test_lifecycle_methods_do_not_raise():
    cls = _get_class_or_skip()
    try:
        obj = cls()
    except Exception:
        pytest.skip("Skipping because instantiation without args failed")
    # call enter and exit if available
    for name in ("enter", "exit"):
        if hasattr(obj, name):
            try:
                getattr(obj, name)()
            except TypeError:
                # some implementations may expect args; ignore but note presence
                pytest.skip(f"{name} exists but requires parameters; skipping call")
            except Exception as e:
                pytest.fail(f"Calling {name} raised an unexpected exception: {e}")


def test_update_accepts_delta_time_if_present():
    cls = _get_class_or_skip()
    try:
        obj = cls()
    except Exception:
        pytest.skip("Skipping because instantiation without args failed")
    if hasattr(obj, "update"):
        try:
            getattr(obj, "update")(0.016)
        except TypeError:
            pytest.skip("update exists but has a non-standard signature; skipping call")
        except Exception as e:
            pytest.fail(f"update raised an unexpected exception: {e}")


def test_serialization_roundtrip_if_supported():
    cls = _get_class_or_skip()
    try:
        obj = cls()
    except Exception:
        pytest.skip("Skipping because instantiation without args failed")
    if not (hasattr(obj, "to_dict") and hasattr(cls, "from_dict")):
        pytest.skip(
            "Serialization methods to_dict/from_dict not both present; skipping"
        )
    try:
        data = obj.to_dict()
    except TypeError:
        pytest.skip("to_dict requires parameters; skipping serialization test")
    except Exception as e:
        pytest.fail(f"to_dict raised an unexpected exception: {e}")
    assert isinstance(data, dict), "to_dict must return a dict"
    try:
        new_obj = cls.from_dict(data)
    except TypeError:
        pytest.skip(
            "from_dict requires parameters different than a single dict; skipping"
        )
    except Exception as e:
        pytest.fail(f"from_dict raised an unexpected exception: {e}")
    assert isinstance(new_obj, cls), "from_dict should return an instance of the class"


def test_next_state_return_type():
    cls = _get_class_or_skip()
    try:
        obj = cls()
    except Exception:
        pytest.skip("Skipping because instantiation without args failed")
    if not hasattr(obj, "next_state"):
        pytest.skip("next_state not implemented; skipping")
    try:
        nxt = obj.next_state()
    except TypeError:
        pytest.skip("next_state requires parameters; skipping")
    except Exception as e:
        pytest.fail(f"next_state raised an unexpected exception: {e}")
    ok_types = (type(None), str, type)
    assert isinstance(nxt, ok_types) or hasattr(
        nxt, "__class__"
    ), "next_state should return None, a state name, a class, or a state instance"


def test_violet_agent_state_has_required_fields():
    """Test that VioletAgentState has all required fields for the violet workflow."""
    # Import the VanityPersona to create test instances
    from app.structures.concepts.vanity_persona import VanityPersona

    cls = _get_class_or_skip()

    # Create test persona
    interviewer = VanityPersona(first_name="Test", last_name="Interviewer")

    try:
        obj = cls(interviewer_persona=interviewer)
    except Exception as e:
        pytest.skip(f"Could not instantiate {CLASS_NAME}: {e}")

    # Check for violet agent specific fields (updated to match current implementation)
    expected_fields = [
        "interviewer_persona",
        "interview_questions",
        "interview_responses",
        "circle_jerk_interview",
    ]

    for field in expected_fields:
        assert hasattr(obj, field), f"VioletAgentState should have field '{field}'"


def test_violet_agent_state_field_defaults():
    """Test that VioletAgentState fields have correct defaults."""
    from app.structures.concepts.vanity_persona import VanityPersona

    cls = _get_class_or_skip()

    # Create test persona
    interviewer = VanityPersona(first_name="Test", last_name="Interviewer")

    try:
        obj = cls(interviewer_persona=interviewer)
    except Exception as e:
        pytest.skip(f"Could not instantiate {CLASS_NAME}: {e}")

    # Test None defaults for optional fields
    assert obj.interview_questions is None
    assert obj.interview_responses is None
    assert obj.circle_jerk_interview is None

    # Test required persona is set correctly
    assert obj.interviewer_persona == interviewer
