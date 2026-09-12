"""
crop_info.py
------------
Short, general educational descriptions for each crop the model can predict.

IMPORTANT: These descriptions are general agricultural knowledge written for
this app. They are NOT produced by the ML model — the model only predicts a
crop name. We keep this clearly separate so we never claim the model itself
"knows" agronomy facts.
"""

CROP_INFO = {
    "rice": {
        "emoji": "🌾",
        "description": "A staple cereal crop grown widely in flooded (paddy) fields.",
        "growing_conditions": "Needs warm temperatures, high humidity, and plenty of water/rainfall. Thrives in clayey soil that holds water well.",
    },
    "maize": {
        "emoji": "🌽",
        "description": "Also known as corn — a versatile cereal crop used for food, feed, and industry.",
        "growing_conditions": "Prefers warm weather, moderate rainfall, and well-drained fertile soil rich in nitrogen.",
    },
    "chickpea": {
        "emoji": "🌱",
        "description": "A protein-rich legume (also called garbanzo bean), popular in many cuisines.",
        "growing_conditions": "Grows best in cool, dry climates with well-drained soil and moderate rainfall.",
    },
    "kidneybeans": {
        "emoji": "🫘",
        "description": "A nutritious legume grown for its protein-rich beans.",
        "growing_conditions": "Prefers moderate temperatures, well-drained loamy soil, and consistent moisture.",
    },
    "pigeonpeas": {
        "emoji": "🌿",
        "description": "A drought-tolerant legume widely grown in semi-arid regions.",
        "growing_conditions": "Tolerates heat and lower rainfall; grows well in well-drained sandy-loam soils.",
    },
    "mothbeans": {
        "emoji": "🌱",
        "description": "A hardy, drought-resistant legume often grown in arid regions.",
        "growing_conditions": "Thrives in hot, dry climates with sandy soil and minimal rainfall.",
    },
    "mungbean": {
        "emoji": "🌱",
        "description": "A fast-growing legume, also known as green gram, rich in protein.",
        "growing_conditions": "Prefers warm temperatures, moderate rainfall, and well-drained soil.",
    },
    "blackgram": {
        "emoji": "🌱",
        "description": "A legume crop (also called urad) commonly used in South Asian cuisine.",
        "growing_conditions": "Grows well in warm, humid conditions with well-drained fertile soil.",
    },
    "lentil": {
        "emoji": "🫘",
        "description": "A small, protein-rich legume grown for its edible seeds.",
        "growing_conditions": "Prefers cool growing seasons, moderate rainfall, and well-drained soil.",
    },
    "pomegranate": {
        "emoji": "🍎",
        "description": "A fruit-bearing shrub known for its juicy, seed-filled fruit.",
        "growing_conditions": "Thrives in hot, dry climates with well-drained soil; tolerates drought once established.",
    },
    "banana": {
        "emoji": "🍌",
        "description": "A fast-growing tropical fruit crop, a major food source worldwide.",
        "growing_conditions": "Needs warm temperatures, high humidity, and consistent rainfall or irrigation.",
    },
    "mango": {
        "emoji": "🥭",
        "description": "A popular tropical fruit tree valued for its sweet fruit.",
        "growing_conditions": "Prefers warm, frost-free climates with a distinct dry season before flowering.",
    },
    "grapes": {
        "emoji": "🍇",
        "description": "A fruit vine grown for fresh consumption, raisins, and wine production.",
        "growing_conditions": "Prefers warm, sunny climates with well-drained soil and moderate rainfall.",
    },
    "watermelon": {
        "emoji": "🍉",
        "description": "A warm-season fruit crop grown for its large, juicy fruit.",
        "growing_conditions": "Needs warm temperatures, plenty of sunlight, and well-drained sandy-loam soil.",
    },
    "muskmelon": {
        "emoji": "🍈",
        "description": "A warm-season fruit closely related to cantaloupe.",
        "growing_conditions": "Prefers hot, dry climates with well-drained soil and moderate irrigation.",
    },
    "apple": {
        "emoji": "🍎",
        "description": "A temperate fruit tree grown for its widely consumed fruit.",
        "growing_conditions": "Needs a cool climate with a distinct winter chilling period and well-drained soil.",
    },
    "orange": {
        "emoji": "🍊",
        "description": "A citrus fruit tree valued for its juicy, vitamin-C-rich fruit.",
        "growing_conditions": "Prefers warm, subtropical climates with moderate rainfall and well-drained soil.",
    },
    "papaya": {
        "emoji": "🍈",
        "description": "A fast-growing tropical fruit tree that bears fruit year-round in warm climates.",
        "growing_conditions": "Needs warm temperatures, high humidity, and well-drained fertile soil.",
    },
    "coconut": {
        "emoji": "🥥",
        "description": "A tropical palm tree grown for its versatile fruit used in food and oil production.",
        "growing_conditions": "Thrives in coastal, humid tropical climates with sandy soil and high rainfall.",
    },
    "cotton": {
        "emoji": "🌱",
        "description": "A major fiber crop grown for textile production.",
        "growing_conditions": "Prefers warm temperatures, moderate rainfall, and a long frost-free growing season.",
    },
    "jute": {
        "emoji": "🌿",
        "description": "A fiber crop used to make ropes, sacks, and other coarse textiles.",
        "growing_conditions": "Needs warm, humid conditions with high rainfall and fertile alluvial soil.",
    },
    "coffee": {
        "emoji": "☕",
        "description": "A tropical shrub grown for its beans, used to produce the popular beverage.",
        "growing_conditions": "Prefers moderate temperatures, high humidity, and well-drained, shaded highland soil.",
    },
}


def get_crop_info(crop_name: str) -> dict:
    """
    Return the educational info dict for a crop, or a safe fallback
    if the crop isn't in our dictionary (keeps the app from crashing).
    """
    key = crop_name.strip().lower()
    return CROP_INFO.get(
        key,
        {
            "emoji": "🌱",
            "description": "General crop information is not available for this crop yet.",
            "growing_conditions": "Please consult local agricultural resources for detailed guidance.",
        },
    )
