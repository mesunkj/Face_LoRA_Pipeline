import json

with open('colab_runner.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "⚠️ 警告：無法在原圖 {os.path.basename(orig_path)} 中偵測到人臉" in line:
                indent = line.split("print")[0]
                new_source.append(f'{indent}print(f"⚠️ 警告：無法在原圖 (檔名: {{os.path.basename(orig_path)}}) 中偵測到人臉，已跳過該 Ground Truth 圖片的比對。")\n')
            elif "⚠️ 警告：找不到場景 '{scene}' 的標靶圖" in line:
                indent = line.split("print")[0]
                new_source.append(f'{indent}print(f"⚠️ 警告：找不到場景 \'{{scene}}\' 的標靶圖，跳過此場景所有 Inpainting 測試 (無產出圖檔代號)！")\n')
            elif "⚠️ 警告：產出圖片 {os.path.basename(gen_path) if 'gen_path' in locals() else ''} 未偵測到人臉" in line:
                indent = line.split("print")[0]
                new_source.append(f'{indent}print(f"⚠️ 警告：產出圖片代號 [{{os.path.splitext(os.path.basename(gen_path))[0]}}] (檔名: {{os.path.basename(gen_path)}}) 未偵測到人臉，給予 0 分。")\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('colab_runner.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
