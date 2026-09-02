import json

s = json.load(open("corpus/data/stats.json", encoding="utf-8"))
tot = 0
for k, v in s["sources"].items():
    tot += v["blocks"]
    print("%-55s %8d  %5.1f%%" % (k, v["blocks"], v["blocks"] / s["blocks"] * 100))
print("%-55s %8d" % ("TOTAL", s["blocks"]))
print("consistent:", tot == s["blocks"])