
import json
import os


DIR_PATH = "./results/KE"


if not os.path.exists("./results/KE/processed"):
    os.makedirs("./results/KE/processed")


for file in os.scandir(DIR_PATH):

    if not file.is_file or not file.name.endswith(".json"):
        continue

    # Read data as json
    with open(file.path) as f:
        data = json.load(f)

    # Collect field names
    fields = [k for k, e in data[0].items() if not isinstance(e, dict)]
    fields += sum(
        [
            list(e.keys())
            for e in data[0].values()
            if isinstance(e, dict)
        ],
        [],
    )

    # Write test results
    with open(file.path.replace(".json", ".csv"), "w") as f:
        f.write(";".join(fields) + "\n")

        for test in data:
            line = [str(e) for e in test.values() if not isinstance(e, dict)]
            line += sum(
                [
                    list(map(str, e.values()))
                    for e in test.values()
                    if isinstance(e, dict)
                ],
                []
            )
            f.write(";".join(line) + "\n")
    
    os.rename(
        file.path,
        file.path.replace("KE/", "KE/processed/")
    )
