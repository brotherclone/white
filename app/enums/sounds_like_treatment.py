from enum import Enum


class SoundsLikeTreatment(Enum):
    nothing = "do nothing"
    remove_b = "remove b"
    b_to_a_remove_b = "b to a, remove b"
    swap_a = "swap a"
    swap_b = "swap b"
    swap_both_a_and_b = "swap both a and b"
    change_location = "change location"
    remove_location = "remove location"
    change_descriptor_a = "change descriptor a"
    change_descriptor_b = "change descriptor b"
    change_descriptor_a_and_b = "change descriptor a and b"
    change_descriptor_a_and_location = "change descriptor a and location"
    change_descriptor_b_and_location = "change descriptor b and location"
    change_descriptor_a_b_and_location = "change descriptor a, b and location"
    remove_descriptor_a = "remove descriptor a"
    remove_descriptor_b = "remove descriptor b"
    remove_descriptor_a_and_b = "remove descriptor a and b"


PLAN_CHANGE_TABLE = [
    (44.0, SoundsLikeTreatment.nothing),
    (45.0, SoundsLikeTreatment.remove_b),
    (50.0, SoundsLikeTreatment.b_to_a_remove_b),
    (54.0, SoundsLikeTreatment.swap_a),
    (58.0, SoundsLikeTreatment.swap_b),
    (62.0, SoundsLikeTreatment.swap_both_a_and_b),
    (66.0, SoundsLikeTreatment.change_location),
    (70.0, SoundsLikeTreatment.remove_location),
    (74.0, SoundsLikeTreatment.change_descriptor_a),
    (77.0, SoundsLikeTreatment.change_descriptor_b),
    (80.0, SoundsLikeTreatment.change_descriptor_a_and_b),
    (83.0, SoundsLikeTreatment.change_descriptor_a_and_location),
    (86.0, SoundsLikeTreatment.change_descriptor_b_and_location),
    (89.0, SoundsLikeTreatment.change_descriptor_a_b_and_location),
    (93.0, SoundsLikeTreatment.remove_descriptor_a),
    (96.0, SoundsLikeTreatment.remove_descriptor_b),
    (99.0, SoundsLikeTreatment.remove_descriptor_a_and_b)
]
