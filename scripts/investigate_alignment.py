import json

arts = {r['id']: r for r in [json.loads(x) for x in open('data/processed/pilot_cleaned.jsonl', encoding='utf-8')]}
flagged = ['141231_rh_bd_online_commerce_', 'news-54145366', 'news-55400029', 'news-39804917']

for fid in flagged:
    a = arts[fid]
    print("=" * 60)
    print("ID:", fid)
    print("TITLE:", a.get("title"))
    print("SUMMARY (Reference):", a.get("summary"))
    print("ARTICLE TEXT (First 350 chars):")
    print(a.get("text")[:350])
    print()
