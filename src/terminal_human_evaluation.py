from pathlib import Path
import pandas as pd


FORM_PATH = Path(
    "outputs/tables/research_human_evaluation_filled.csv"
)


REQUIRED_COLUMNS = [
    "best_system",
    "adequacy_a_1_to_5",
    "adequacy_b_1_to_5",
    "adequacy_c_1_to_5",
    "fluency_a_1_to_5",
    "fluency_b_1_to_5",
    "fluency_c_1_to_5",
]


ERROR_COLUMNS = [
    "missing_information_a",
    "missing_information_b",
    "missing_information_c",
    "hallucination_a",
    "hallucination_b",
    "hallucination_c",
    "named_entity_error_a",
    "named_entity_error_b",
    "named_entity_error_c",
]


def save_file(df):
    df.to_csv(
        FORM_PATH,
        index=False,
        encoding="utf-8-sig"
    )


def is_completed(row):
    for column in REQUIRED_COLUMNS:
        value = row.get(column)

        if pd.isna(value):
            return False

        if str(value).strip() == "":
            return False

    return True


def ask_best_system():
    while True:
        value = input(
            "\nBest system [A/B/C], S to skip, Q to quit: "
        ).strip().upper()

        if value in {"A", "B", "C", "S", "Q"}:
            return value

        print("Enter only A, B, C, S, or Q.")


def ask_three_scores(title):
    while True:
        value = input(
            f"{title} for A B C, for example 3 5 4: "
        ).strip()

        if value.upper() == "Q":
            return "Q"

        parts = value.replace(",", " ").split()

        if len(parts) != 3:
            print("Enter exactly three scores.")
            continue

        try:
            scores = [int(score) for score in parts]
        except ValueError:
            print("Scores must be numbers from 1 to 5.")
            continue

        if all(1 <= score <= 5 for score in scores):
            return scores

        print("Every score must be between 1 and 5.")


def ask_three_errors(title):
    while True:
        value = input(
            f"{title} for A B C [Y/N/U], example N N Y: "
        ).strip().upper()

        if value == "Q":
            return "Q"

        parts = value.replace(",", " ").split()

        if len(parts) != 3:
            print("Enter exactly three values.")
            continue

        if all(item in {"Y", "N", "U"} for item in parts):
            conversion = {
                "Y": "Yes",
                "N": "No",
                "U": "",
            }

            return [
                conversion[item]
                for item in parts
            ]

        print(
            "Use Y for Yes, N for No, "
            "or U when you are unsure."
        )


def show_sample(row, current, total):
    print("\n")
    print("=" * 110)
    print(f"SAMPLE {current} OF {total}")
    print("=" * 110)

    print("\nPASHTO SOURCE")
    print("-" * 110)
    print(str(row.get("pashto_source", "")))

    print("\nENGLISH REFERENCE")
    print("-" * 110)
    print(str(row.get("english_reference", "")))

    print("\nSYSTEM A")
    print("-" * 110)
    print(str(row.get("system_a_output", "")))

    print("\nSYSTEM B")
    print("-" * 110)
    print(str(row.get("system_b_output", "")))

    print("\nSYSTEM C")
    print("-" * 110)
    print(str(row.get("system_c_output", "")))

    print("\nSCORING SCALE")
    print("-" * 110)
    print("1 = completely incorrect")
    print("2 = mostly incorrect")
    print("3 = partially correct")
    print("4 = mostly correct")
    print("5 = fully correct")


def main():
    if not FORM_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation form not found: {FORM_PATH}"
        )

    df = pd.read_csv(
        FORM_PATH,
        encoding="utf-8-sig"
    )

    for column in REQUIRED_COLUMNS + ERROR_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    text_columns = [
        "best_system",
        *ERROR_COLUMNS,
        "comments",
    ]

    for column in text_columns:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].astype("object")

    score_columns = [
        "adequacy_a_1_to_5",
        "adequacy_b_1_to_5",
        "adequacy_c_1_to_5",
        "fluency_a_1_to_5",
        "fluency_b_1_to_5",
        "fluency_c_1_to_5",
    ]

    for column in score_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    completed_before = sum(
        is_completed(row)
        for _, row in df.iterrows()
    )

    print("=" * 80)
    print("TERMINAL HUMAN EVALUATION")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Already completed: {completed_before}")
    print()
    print("The file is saved automatically after every sample.")
    print("Enter Q at any prompt to save and stop.")
    print("Run the same program later to continue.")

    for index, row in df.iterrows():
        if is_completed(row):
            continue

        show_sample(
            row,
            index + 1,
            len(df)
        )

        best_system = ask_best_system()

        if best_system == "Q":
            save_file(df)
            print("\nProgress saved.")
            return

        if best_system == "S":
            continue

        adequacy = ask_three_scores(
            "Adequacy scores"
        )

        if adequacy == "Q":
            save_file(df)
            print("\nProgress saved.")
            return

        fluency = ask_three_scores(
            "Fluency scores"
        )

        if fluency == "Q":
            save_file(df)
            print("\nProgress saved.")
            return

        missing_information = ask_three_errors(
            "Missing information"
        )

        if missing_information == "Q":
            save_file(df)
            print("\nProgress saved.")
            return

        hallucination = ask_three_errors(
            "Hallucination"
        )

        if hallucination == "Q":
            save_file(df)
            print("\nProgress saved.")
            return

        named_entity_error = ask_three_errors(
            "Named-entity error"
        )

        if named_entity_error == "Q":
            save_file(df)
            print("\nProgress saved.")
            return

        df.at[index, "best_system"] = best_system

        df.at[index, "adequacy_a_1_to_5"] = adequacy[0]
        df.at[index, "adequacy_b_1_to_5"] = adequacy[1]
        df.at[index, "adequacy_c_1_to_5"] = adequacy[2]

        df.at[index, "fluency_a_1_to_5"] = fluency[0]
        df.at[index, "fluency_b_1_to_5"] = fluency[1]
        df.at[index, "fluency_c_1_to_5"] = fluency[2]

        df.at[index, "missing_information_a"] = (
            missing_information[0]
        )

        df.at[index, "missing_information_b"] = (
            missing_information[1]
        )

        df.at[index, "missing_information_c"] = (
            missing_information[2]
        )

        df.at[index, "hallucination_a"] = hallucination[0]
        df.at[index, "hallucination_b"] = hallucination[1]
        df.at[index, "hallucination_c"] = hallucination[2]

        df.at[index, "named_entity_error_a"] = (
            named_entity_error[0]
        )

        df.at[index, "named_entity_error_b"] = (
            named_entity_error[1]
        )

        df.at[index, "named_entity_error_c"] = (
            named_entity_error[2]
        )

        save_file(df)

        completed_now = sum(
            is_completed(current_row)
            for _, current_row in df.iterrows()
        )

        print(
            f"\nSaved successfully. "
            f"Completed: {completed_now}/{len(df)}"
        )

    save_file(df)

    completed_final = sum(
        is_completed(row)
        for _, row in df.iterrows()
    )

    print("\n" + "=" * 80)
    print("EVALUATION FINISHED")
    print("=" * 80)
    print(f"Completed samples: {completed_final}")
    print(f"Saved file: {FORM_PATH}")


if __name__ == "__main__":
    main()
