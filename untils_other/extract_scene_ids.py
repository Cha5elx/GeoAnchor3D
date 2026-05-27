import json

def extract_scene_ids(json_file):
    """从JSON文件中提取所有唯一的scene_id"""
    scene_ids = set()
    
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 提取所有scene_id
    for item in data:
        if 'scene_id' in item:
            scene_ids.add(item['scene_id'])
    
    print(f"已从 {json_file} 提取 {len(scene_ids)} 个唯一的scene_id")
    return scene_ids

# 处理三个文件
files_to_process = [
    "/home/lcx/chat-scene/Chat-Scene/annotations/scanqa_train_mini_50pct.json",
    "/home/lcx/chat-scene/Chat-Scene/annotations/scanrefer_mask3d_train_mini_50pct.json",
    "/home/lcx/chat-scene/Chat-Scene/annotations/obj_align_mask3d_train_mini_50pct.json"
]

all_scene_ids = set()

for json_file in files_to_process:
    scene_ids = extract_scene_ids(json_file)
    all_scene_ids.update(scene_ids)

# 计算三个文件的并集
union_scene_ids = sorted(all_scene_ids)
union_output_file = "/home/lcx/chat-scene/Chat-Scene/untils_other/50pct_scene_ids_union.txt"

# 写入并集到txt文件
with open(union_output_file, 'w') as f:
    for scene_id in union_scene_ids:
        f.write(f"{scene_id}\n")

print(f"已将三个文件的并集（共{len(union_scene_ids)}个唯一的scene_id）保存到 {union_output_file}")
print("所有文件处理完成！")