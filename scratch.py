import requests

def check_hf(repo):
    url = f"https://huggingface.co/api/datasets/{repo}"
    r = requests.get(url)
    print(f"{repo}: {r.status_code}")

check_hf("THUDM/RGB")
check_hf("chenghao/rgb")
check_hf("hotpot_qa")
