import json


def load_json_or_jsonl(filename):

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try normal JSON first
    try:
        return json.loads(content)

    except json.JSONDecodeError:

        # If normal JSON fails, try JSONL
        data = []

        for line_number, line in enumerate(content.splitlines(), start=1):

            line = line.strip()

            if not line:
                continue

            try:
                data.append(json.loads(line))

            except json.JSONDecodeError as e:
                print(f"Error on line {line_number}: {e}")
                raise

        return data


def convert_json_to_text(data, output_file="evaluation_results.txt"):

    if isinstance(data, dict):
        data = [data]

    with open(output_file, "w", encoding="utf-8") as f:

        for i, item in enumerate(data, start=1):

            f.write("=" * 100 + "\n")
            f.write(f"CONSULTATION {i}\n")
            f.write("=" * 100 + "\n\n")

            # --------------------------------------------------
            # CONSULTATION
            # --------------------------------------------------

            f.write("### CONSULTATION\n\n")
            f.write(item.get("consultation", "N/A"))
            f.write("\n\n")

            # --------------------------------------------------
            # REPORT
            # --------------------------------------------------

            f.write("=" * 100 + "\n")
            f.write("### GENERATED CLINICAL REPORT\n")
            f.write("=" * 100 + "\n\n")

            f.write(item.get("report", "N/A"))
            f.write("\n\n")

            # --------------------------------------------------
            # FACTUALITY SCORES
            # --------------------------------------------------

            f.write("=" * 100 + "\n")
            f.write("### FACTUALITY SCORES\n")
            f.write("=" * 100 + "\n\n")

            scores = item.get("factuality_scores", {})

            f.write(
                f"Patient Factuality Score: "
                f"{scores.get('patient_factuality_score', 'N/A')}\n"
            )

            f.write(
                f"Syndrome Factuality Score: "
                f"{scores.get('syndrome_factuality_score', 'N/A')}\n"
            )

            f.write(
                f"Average Factuality Score: "
                f"{scores.get('average_factuality_score', 'N/A')}\n"
            )

            f.write("\n")

            # --------------------------------------------------
            # CLAIM SUMMARY
            # --------------------------------------------------

            f.write("=" * 100 + "\n")
            f.write("### CLAIM SUMMARY\n")
            f.write("=" * 100 + "\n\n")

            counts = item.get("summary_counts", {})

            f.write(
                f"Total Patient Claims: "
                f"{counts.get('patient_claims', 'N/A')}\n"
            )

            f.write(
                f"Supported Patient Claims: "
                f"{counts.get('patient_supported', 'N/A')}\n"
            )

            f.write(
                f"Total Syndrome Claims: "
                f"{counts.get('syndrome_claims', 'N/A')}\n"
            )

            f.write(
                f"Supported Syndrome Claims: "
                f"{counts.get('syndrome_supported', 'N/A')}\n"
            )

            f.write("\n")

            # --------------------------------------------------
            # PATIENT CLAIM AUDIT
            # --------------------------------------------------

            f.write("=" * 100 + "\n")
            f.write("### PATIENT-SPECIFIC CLAIM AUDIT\n")
            f.write("=" * 100 + "\n\n")

            audit = item.get("audit_breakdown", {})
            patient_checks = audit.get("patient_claims_check", [])

            patient_yes = 0
            patient_no = 0

            for j, claim_data in enumerate(patient_checks, start=1):

                claim = claim_data.get("claim", "")
                supported = claim_data.get("supported", "UNKNOWN")

                f.write(f"{j}. {claim}\n")
                f.write(f"   Supported: {supported}\n\n")

                if supported.upper() == "YES":
                    patient_yes += 1

                elif supported.upper() == "NO":
                    patient_no += 1

            f.write(f"Patient YES: {patient_yes}\n")
            f.write(f"Patient NO: {patient_no}\n\n")

            # --------------------------------------------------
            # SYNDROME CLAIM AUDIT
            # --------------------------------------------------

            f.write("=" * 100 + "\n")
            f.write("### SYNDROME-SPECIFIC CLAIM AUDIT\n")
            f.write("=" * 100 + "\n\n")

            syndrome_checks = audit.get("syndrome_claims_check", [])

            syndrome_yes = 0
            syndrome_no = 0

            for j, claim_data in enumerate(syndrome_checks, start=1):

                claim = claim_data.get("claim", "")
                supported = claim_data.get("supported", "UNKNOWN")

                f.write(f"{j}. {claim}\n")
                f.write(f"   Supported: {supported}\n\n")

                if supported.upper() == "YES":
                    syndrome_yes += 1

                elif supported.upper() == "NO":
                    syndrome_no += 1

            f.write(f"Syndrome YES: {syndrome_yes}\n")
            f.write(f"Syndrome NO: {syndrome_no}\n\n")

            # --------------------------------------------------
            # END OF RECORD
            # --------------------------------------------------

            f.write("\n\n")
            
            if i ==4:
                break


# ============================================================
# MAIN
# ============================================================

input_file = r"C:\Users\Aman Kumar Singh\Desktop\nlp_dissertation\Distillation\baseline_qwen25_15b\factuality_scores_[baseline_qwen25_15b]_01.jsonl"
output_file = "evaluation_results_base.txt"

data = load_json_or_jsonl(input_file)

convert_json_to_text(data, output_file)

print("Done!")
print(f"Processed {len(data)} records.")
print(f"Output saved to: {output_file}")