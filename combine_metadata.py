import os,json
all_metadata=dict()
for folder in os.listdir(".."):
    if folder.startswith("dfdc"):
        metadata=json.loads(open(folder+"/metadata.json","r"))
        all_metadata.update(metadata)
json.dump(all_metadata,open("all_metadata.json","w"))