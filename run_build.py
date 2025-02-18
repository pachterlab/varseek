import os
import subprocess
out = "/Users/joeyrich/Desktop/local/varseek/data/vk_build_test13"
os.makedirs(out, exist_ok=True)
subprocess.run(f"cp -r /Users/joeyrich/Desktop/local/varseek/data/vk_build_test10/reference {out}/", shell=True, check=True)

import varseek as vk
vk.varseek_build.build(
    variants="cosmic_cmc",
    sequences="cdna",
    out=out,
    gtf=True,  # just so that gtf information gets merged into cosmic df
)