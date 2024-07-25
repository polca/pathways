import bw2data
import json

bw2data.projects.set_current("ei310")

lcia = []
for m in bw2data.methods:
    print(m)
    method = bw2data.Method(m)

    dm = {
        "name": list(method.name),
        "unit": method.metadata["unit"],

    }
    exc = []
    for name, cf in method.load():
        act = bw2data.get_activity(name)
        name = act["name"]
        cat = list(act["categories"])
        exc.append(
            {
                "name": name,
                "categories": cat,
                "amount": cf
            }
        )
    dm["exchanges"] = exc
    lcia.append(dm)


with open("lcia_ei310.json", "w") as file:
    json.dump(lcia, file)