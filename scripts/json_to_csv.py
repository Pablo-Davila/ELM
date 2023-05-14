
import json
import sys


SOURCE_PATH = sys.argv[1]
TARGET_PATH = SOURCE_PATH.replace(".json", ".csv")


# Read data as json
with open(SOURCE_PATH) as f:
    data = json.load(f)

# Collect field names
fields = [k for k, e in data[0].items() if not isinstance(e, dict)]
fields += sum([list(e.keys())
              for e in data[0].values() if isinstance(e, dict)], [])

# Write test results
with open(TARGET_PATH, "w") as f:
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
