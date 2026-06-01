from owlready2 import *

# ------------------------------------------------------------
# Load ontology
# ------------------------------------------------------------

onto_path.append(".")

ONTOLOGY_FILE = "road_sign_ontology_Sara_Maria_Gussem.owl"
onto = get_ontology(ONTOLOGY_FILE).load()

print("Ontology loaded successfully:", ONTOLOGY_FILE)

# ------------------------------------------------------------
# Optional reasoner
# ------------------------------------------------------------
# Pellet requires Java. If Java is not installed, the script continues
# using asserted ontology axioms only.

RUN_REASONER = False

if RUN_REASONER:
    try:
        with onto:
            sync_reasoner_pellet(
                infer_property_values=True,
                infer_data_property_values=True
            )
        print("Pellet reasoning completed successfully.\n")
    except Exception as e:
        print("Pellet reasoning could not be completed.")
        print("Continuing with asserted ontology axioms only.\n")
        print("Reasoner error:", e, "\n")
else:
    print("Reasoner skipped in Python. Using asserted ontology axioms only.\n")


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def readable(entity):
    """Return a readable name for OWL entities."""
    if entity is None:
        return "None"
    if hasattr(entity, "name"):
        return entity.name
    return str(entity)


def restriction_fillers_for_property(cls, prop):
    """
    Return fillers from restrictions directly asserted on a class.

    Handles:
    - prop some X
    - prop only X
    - prop value X
    """
    fillers = set()

    axioms_to_check = []
    axioms_to_check.extend(list(cls.is_a))
    axioms_to_check.extend(list(cls.equivalent_to))

    for ax in axioms_to_check:
        if isinstance(ax, Restriction) and ax.property == prop:
            if hasattr(ax, "value") and ax.value is not None:
                fillers.add(ax.value)

    return fillers


def get_shapes_for_category(category, shape_property):
    """
    Return shapes associated with a category or its subclasses.
    """
    shapes = set()

    if category is None:
        return shapes

    for cls in category.descendants(include_self=True):
        shapes.update(restriction_fillers_for_property(cls, shape_property))

    return shapes


# ------------------------------------------------------------
# Identify shape property
# ------------------------------------------------------------

possible_shape_properties = [
    "hasShape",
    "hasSignShape",
    "hasShapeName"
]

shape_property = None
shape_property_name = None

for prop_name in possible_shape_properties:
    prop = getattr(onto, prop_name, None)
    if prop is not None:
        shape_property = prop
        shape_property_name = prop_name
        break

if shape_property is None:
    print("No shape property found.")
    print("Checked:", possible_shape_properties)
    print("\nAvailable object properties:")
    for prop in onto.object_properties():
        print("-", prop.name)
    print("\nAvailable data properties:")
    for prop in onto.data_properties():
        print("-", prop.name)
    raise SystemExit

print("Shape property used:", shape_property_name, "\n")


# ------------------------------------------------------------
# Traffic sign categories
# ------------------------------------------------------------

candidate_category_names = [
    "DangerWarningSign",
    "WarningSign",
    "RegulatorySign",
    "PrioritySign",
    "ProhibitorySign",
    "ProhibitoryOrRestrictiveSign",
    "MandatorySign",
    "SpecialRegulationSign",
    "InformativeSign",
    "InformationSign",
    "InformationFacilityServiceSign",
    "DirectionSign",
    "DirectionPositionOrIndicationSign",
    "ServiceSign",
    "AdditionalPanel"
]

categories = []

for name in candidate_category_names:
    cls = getattr(onto, name, None)
    if cls is not None:
        categories.append(cls)

if not categories:
    print("No traffic sign categories found. Check class names.")
    raise SystemExit


# ------------------------------------------------------------
# Query: shapes associated with each category
# ------------------------------------------------------------

print("Shapes associated with each traffic sign category:\n")

any_shape_found = False

for category in categories:
    shapes = get_shapes_for_category(category, shape_property)

    print(f"- {category.name}:")

    if shapes:
        any_shape_found = True
        for shape in sorted(shapes, key=lambda x: readable(x)):
            print(f"  • {readable(shape)}")
    else:
        print("  • No shape restriction found")

    print()

if not any_shape_found:
    print("Summary:")
    print("No shape restrictions were found for the selected traffic sign categories.")
    print("The ontology contains the shape property, but it does not appear to use it in class restrictions.")
    print("This means the competency question cannot be answered at class-restriction level with the current ontology.")