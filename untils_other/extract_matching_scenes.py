import json

# 读取scene_id列表
with open('/home/lcx/chat-scene/Chat-Scene/untils_other/50pct_scene_ids_union.txt', 'r') as f:
    scene_ids = set(line.strip() for line in f if line.strip())

print(f'从 scene_ids 文件读取了 {len(scene_ids)} 个scene_id')

# 读取原始JSON文件
with open('/home/lcx/chat-scene/Chat-Scene/annotations/object_descriptions_with_obj_tags.json', 'r') as f:
    data = json.load(f)

# 筛选匹配的记录
filtered_data = []
for item in data:
    if item.get('scene_id') in scene_ids:
        filtered_data.append(item)

print(f'从原始JSON文件中筛选出 {len(filtered_data)} 条记录')

# 保存筛选后的数据
with open('/home/lcx/chat-scene/Chat-Scene/annotations/object_descriptions_with_obj_tags_mini_50pct.json', 'w') as f:
    json.dump(filtered_data, f, indent=2)

print('已将筛选后的数据保存到 object_descriptions_with_obj_tags_mini_50pct.json')
print('处理完成！')