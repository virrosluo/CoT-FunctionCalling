from Function import viettelFunc
import json

output = viettelFunc.CSDL_TRAM("HN1028")

print(json.dumps(output, ensure_ascii=False))