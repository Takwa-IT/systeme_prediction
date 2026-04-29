from pathlib import Path
import csv
import random


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "dataset_huilerie_multietapes_5000.csv"


random.seed(42)

HEADERS = [
    "lot_id",
    "variete",
    "region",
    "methode_recolte",
    "type_sol",
    "lavage_effectue",
    "type_machine",
    "type_broyeur",
    "type_malaxeur",
    "type_nettoyage",
    "type_separation",
    "controle_temperature",
    "poids_olives_kg",
    "maturite_niveau_1_5",
    "duree_stockage_jours",
    "temps_depuis_recolte_heures",
    "temperature_malaxage_c",
    "duree_malaxage_min",
    "vitesse_decanteur_tr_min",
    "humidite_pourcent",
    "acidite_olives_pourcent",
    "taux_feuilles_pourcent",
    "pression_extraction_bar",
    "nombre_etapes",
    "presence_ajout_eau",
    "presence_presse",
    "presence_separateur",
    "acidite_huile_pourcent",
    "indice_peroxyde_meq_o2_kg",
    "polyphenols_mg_kg",
    "k232",
    "k270",
    "classe_qualite",
    "rendement_extraction_pourcent",
]

VARIETES = ["Chemlali", "Chetoui", "Arbequina"]
REGIONS = ["Nord", "Centre", "Sud"]
METHODES = ["manuelle", "mecanique", "semi-mecanique"]
SOLS = ["calcaire", "argileux", "sableux"]
MACHINES = [
    {
        "type_machine": "3_phase",
        "type_broyeur": "marteaux",
        "type_malaxeur": "vertical",
        "type_nettoyage": "soufflerie",
        "type_separation": "decanteur_3_phases",
        "controle_temperature": "oui",
        "nombre_etapes": 7,
        "presence_ajout_eau": 1,
        "presence_presse": 0,
        "presence_separateur": 1,
        "pression_extraction_bar": 95,
    },
    {
        "type_machine": "2_phase",
        "type_broyeur": "marteaux",
        "type_malaxeur": "horizontal",
        "type_nettoyage": "laveuse_eau",
        "type_separation": "decanteur_2_phases",
        "controle_temperature": "oui",
        "nombre_etapes": 6,
        "presence_ajout_eau": 0,
        "presence_presse": 0,
        "presence_separateur": 0,
        "pression_extraction_bar": 105,
    },
    {
        "type_machine": "presse",
        "type_broyeur": "meule",
        "type_malaxeur": "horizontal",
        "type_nettoyage": "separateur_feuilles",
        "type_separation": "decantation_naturelle",
        "controle_temperature": "non",
        "nombre_etapes": 6,
        "presence_ajout_eau": 0,
        "presence_presse": 1,
        "presence_separateur": 1,
        "pression_extraction_bar": 145,
    },
]

VARIETY_WEIGHTS_BY_REGION = {
    "Nord": [("Chetoui", 5), ("Chemlali", 3), ("Arbequina", 2)],
    "Centre": [("Chemlali", 5), ("Chetoui", 2), ("Arbequina", 1)],
    "Sud": [("Chemlali", 5), ("Arbequina", 3), ("Chetoui", 1)],
}

REGION_PROFILES = {
    "Nord": {"acidite": 0.33, "polyphenols": 430, "rendement": 19.2},
    "Centre": {"acidite": 0.4, "polyphenols": 370, "rendement": 18.3},
    "Sud": {"acidite": 0.46, "polyphenols": 325, "rendement": 17.6},
}

VARIETY_PROFILES = {
    "Chemlali": {"acidite_delta": -0.02, "polyphenols_delta": 12, "rendement_delta": 0.25},
    "Chetoui": {"acidite_delta": -0.05, "polyphenols_delta": 28, "rendement_delta": 0.15},
    "Arbequina": {"acidite_delta": 0.0, "polyphenols_delta": 6, "rendement_delta": -0.1},
}


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def weighted_choice(options):
    total = sum(weight for _, weight in options)
    pick = random.uniform(0, total)
    cumulative = 0
    for value, weight in options:
        cumulative += weight
        if pick <= cumulative:
            return value
    return options[-1][0]


def quality_from_metrics(acidite_huile, peroxyde, polyphenols, rendement):
    score = 0
    if acidite_huile <= 0.5:
        score += 2
    elif acidite_huile <= 0.8:
        score += 1

    if peroxyde <= 10:
        score += 2
    elif peroxyde <= 15:
        score += 1

    if polyphenols >= 400:
        score += 2
    elif polyphenols >= 300:
        score += 1

    if rendement >= 18:
        score += 1

    if score >= 6:
        return "Excellente"
    if score >= 4:
        return "Bonne"
    return "Moyenne"


def generate_rows(count=500):
    rows = []

    for index in range(1, count + 1):
        machine = random.choice(MACHINES)
        region = random.choice(REGIONS)
        variete = weighted_choice(VARIETY_WEIGHTS_BY_REGION[region])
        methode = random.choice(METHODES)
        sol = random.choice(SOLS)

        region_profile = REGION_PROFILES[region]
        variety_profile = VARIETY_PROFILES[variete]

        poids = round(random.uniform(2200, 12000), 1)
        maturite = random.choices([2, 3, 4, 5], weights=[2, 5, 2, 1], k=1)[0]
        duree_stockage = random.choices([0, 1, 2, 3, 4, 5], weights=[
                                        5, 4, 2, 1, 1, 1], k=1)[0]
        temps_depuis_recolte = round(
            clamp(random.gauss(8.5, 2.8), 3.0, 22.0), 1)
        temperature_malaxage = round(clamp(random.gauss(
            26.2 if machine["type_machine"] != "presse" else 27.0, 0.9), 23.5, 29.0), 1)
        duree_malaxage = int(clamp(random.gauss(
            34 if machine["type_machine"] != "presse" else 38, 4), 24, 48))
        vitesse_decanteur = random.choice(
            [3200, 3250, 3300, 3350, 3380, 3400]) if machine["type_machine"] != "presse" else 0
        humidite = round(clamp(random.gauss(16.8, 1.8), 12.0, 22.5), 1)
        acidite_olives = round(
            clamp(
                random.gauss(
                    region_profile["acidite"] + variety_profile["acidite_delta"], 0.08),
                0.15,
                0.95,
            ),
            2,
        )
        taux_feuilles = round(
            clamp(random.gauss(0.65 if methode == "mecanique" else 0.9, 0.3), 0.1, 2.2), 1)

        if machine["type_machine"] == "3_phase":
            lavage = "oui"
        elif machine["type_machine"] == "2_phase":
            lavage = random.choice(["oui", "oui", "non"])
        else:
            lavage = random.choice(["non", "oui"])

        if machine["type_machine"] == "3_phase":
            methode = random.choices(
                ["mecanique", "semi-mecanique", "manuelle"], weights=[5, 3, 1], k=1)[0]
        elif machine["type_machine"] == "2_phase":
            methode = random.choices(
                ["mecanique", "semi-mecanique", "manuelle"], weights=[4, 4, 1], k=1)[0]
        else:
            methode = random.choices(
                ["manuelle", "semi-mecanique", "mecanique"], weights=[5, 3, 1], k=1)[0]

        if machine["type_machine"] == "presse":
            sol = random.choices(
                ["argileux", "calcaire", "sableux"], weights=[4, 3, 2], k=1)[0]
        elif region == "Nord":
            sol = random.choices(
                ["calcaire", "argileux", "sableux"], weights=[4, 3, 1], k=1)[0]
        elif region == "Centre":
            sol = random.choices(
                ["argileux", "calcaire", "sableux"], weights=[4, 2, 1], k=1)[0]
        else:
            sol = random.choices(
                ["sableux", "argileux", "calcaire"], weights=[4, 2, 1], k=1)[0]

        base_acidite_huile = {
            "3_phase": 0.36,
            "2_phase": 0.43,
            "presse": 0.6,
        }[machine["type_machine"]]
        acidite_huile = round(
            clamp(
                random.gauss(
                    base_acidite_huile +
                    region_profile["acidite"] * 0.04 +
                    variety_profile["acidite_delta"] * 0.35,
                    0.12,
                ),
                0.18,
                1.1,
            ),
            2,
        )
        peroxyde = round(
            clamp(
                random.gauss(
                    7.8 if machine["type_machine"] == "3_phase" else 9.0 if machine["type_machine"] == "2_phase" else 11.2,
                    1.8,
                ),
                4.5,
                16.5,
            ),
            1,
        )
        polyphenols = int(
            clamp(
                random.gauss(
                    region_profile["polyphenols"] + variety_profile["polyphenols_delta"] + (
                        18 if machine["type_machine"] == "3_phase" else 0 if machine["type_machine"] == "2_phase" else -35),
                    55,
                ),
                180,
                620,
            )
        )
        k232 = round(clamp(random.gauss(
            1.55 if machine["type_machine"] != "presse" else 1.78, 0.12), 1.15, 2.2), 2)
        k270 = round(clamp(random.gauss(
            0.11 if machine["type_machine"] == "3_phase" else 0.14 if machine["type_machine"] == "2_phase" else 0.19, 0.03), 0.06, 0.28), 2)

        base_yield = {
            "3_phase": 19.2,
            "2_phase": 17.8,
            "presse": 14.2,
        }[machine["type_machine"]]

        rendement = base_yield
        rendement += (maturite - 3) * 0.6
        rendement -= duree_stockage * 0.35
        rendement -= max(0, temps_depuis_recolte - 8) * 0.18
        rendement += (0.6 - acidite_olives) * 1.4
        rendement += (18 - humidite) * 0.12
        rendement += variety_profile["rendement_delta"]
        rendement += (region_profile["rendement"] - 18.5) * 0.25
        rendement += 0.15 if methode == "mecanique" else 0.0
        rendement += -0.2 if methode == "manuelle" else 0.0
        rendement += random.uniform(-1.2, 1.2)
        rendement = round(clamp(rendement, 12.0, 24.5), 1)

        classe_qualite = quality_from_metrics(
            acidite_huile, peroxyde, polyphenols, rendement)

        rows.append(
            [
                f"L{index:03d}",
                variete,
                region,
                methode,
                sol,
                lavage,
                machine["type_machine"],
                machine["type_broyeur"],
                machine["type_malaxeur"],
                machine["type_nettoyage"],
                machine["type_separation"],
                machine["controle_temperature"],
                poids,
                maturite,
                duree_stockage,
                temps_depuis_recolte,
                temperature_malaxage,
                duree_malaxage,
                vitesse_decanteur,
                humidite,
                acidite_olives,
                taux_feuilles,
                machine["pression_extraction_bar"],
                machine["nombre_etapes"],
                machine["presence_ajout_eau"],
                machine["presence_presse"],
                machine["presence_separateur"],
                acidite_huile,
                peroxyde,
                polyphenols,
                k232,
                k270,
                classe_qualite,
                rendement,
            ]
        )

    return rows


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(5000)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEADERS)
        writer.writerows(rows)

    print(f"CSV généré: {OUTPUT_PATH.resolve()}")
    print(f"Lignes: {len(rows)}")


if __name__ == "__main__":
    main()
