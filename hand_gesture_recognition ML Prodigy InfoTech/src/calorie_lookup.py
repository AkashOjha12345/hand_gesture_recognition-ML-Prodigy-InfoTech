from src.preprocess import DEFAULT_CALORIES, normalize_label

CALORIE_ESTIMATES = {
    "apple_pie": 320,
    "baby_back_ribs": 430,
    "baklava": 290,
    "beef_carpaccio": 180,
    "beef_tartare": 250,
    "beet_salad": 170,
    "beignets": 310,
    "bibimbap": 490,
    "bread_pudding": 360,
    "breakfast_burrito": 520,
    "bruschetta": 180,
    "caesar_salad": 240,
    "cannoli": 260,
    "caprese_salad": 220,
    "carrot_cake": 380,
    "ceviche": 190,
    "cheesecake": 430,
    "cheese_plate": 410,
    "chicken_curry": 420,
    "chicken_quesadilla": 470,
    "chicken_wings": 430,
    "chocolate_cake": 410,
    "chocolate_mousse": 310,
    "churros": 280,
    "clam_chowder": 210,
    "club_sandwich": 480,
    "crab_cakes": 290,
    "creme_brulee": 300,
    "croque_madame": 540,
    "cup_cakes": 260,
    "deviled_eggs": 190,
    "donuts": 290,
    "dumplings": 280,
    "edamame": 140,
    "eggs_benedict": 430,
    "escargots": 170,
    "falafel": 330,
    "filet_mignon": 460,
    "fish_and_chips": 590,
    "foie_gras": 370,
    "french_fries": 365,
    "french_onion_soup": 250,
    "french_toast": 350,
    "fried_calamari": 330,
    "fried_rice": 410,
    "frozen_yogurt": 210,
    "garlic_bread": 230,
    "gnocchi": 360,
    "greek_salad": 210,
    "grilled_cheese_sandwich": 400,
    "grilled_salmon": 370,
    "guacamole": 180,
    "gyoza": 250,
    "hamburger": 520,
    "hot_and_sour_soup": 170,
    "hot_dog": 290,
    "huevos_rancheros": 400,
    "hummus": 170,
    "ice_cream": 270,
    "lasagna": 480,
    "lobster_bisque": 240,
    "lobster_roll_sandwich": 430,
    "macaroni_and_cheese": 450,
    "macarons": 240,
    "miso_soup": 90,
    "mussels": 220,
    "nachos": 490,
    "omelette": 260,
    "onion_rings": 360,
    "oysters": 150,
    "pad_thai": 520,
    "paella": 460,
    "pancakes": 350,
    "panna_cotta": 280,
    "peking_duck": 470,
    "pho": 320,
    "pizza": 430,
    "pork_chop": 410,
    "poutine": 560,
    "prime_rib": 600,
    "pulled_pork_sandwich": 510,
    "ramen": 470,
    "ravioli": 340,
    "red_velvet_cake": 430,
    "risotto": 390,
    "samosa": 260,
    "sashimi": 160,
    "scallops": 240,
    "seaweed_salad": 120,
    "shrimp_and_grits": 430,
    "spaghetti_bolognese": 470,
    "spaghetti_carbonara": 520,
    "spring_rolls": 220,
    "steak": 560,
    "strawberry_shortcake": 330,
    "sushi": 300,
    "tacos": 320,
    "takoyaki": 290,
    "tiramisu": 420,
    "tuna_tartare": 210,
    "waffles": 380,
}


def estimate_calories(label: str, calorie_map: dict[str, int] | None = None) -> int:
    normalized = normalize_label(label)
    if calorie_map and normalized in calorie_map:
        return int(calorie_map[normalized])
    if normalized in CALORIE_ESTIMATES:
        return CALORIE_ESTIMATES[normalized]
    return DEFAULT_CALORIES


def build_calorie_map(class_names: list[str], overrides: dict[str, int] | None = None) -> dict[str, int]:
    mapping: dict[str, int] = {}
    overrides = overrides or {}

    for class_name in class_names:
        normalized = normalize_label(class_name)
        if normalized in overrides:
            mapping[normalized] = int(overrides[normalized])
        else:
            mapping[normalized] = estimate_calories(normalized)

    return mapping
