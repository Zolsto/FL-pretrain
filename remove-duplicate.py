import pandas as pd

# Function to copy one file from one folder to another
def copy_file(source: str, destination: str):
    try:
        with open(source, "rb") as file_s:
            with open(destination, "wb") as file_d:
                file_d.write(file_s.read())
                
    except FileNotFoundError:
        print(f"File not found!")
        
    except Exception as err:
        print(f"Error during copy process: {err}")

dataset = "../../Complete dataset without sbd"
dest = "../../data-noSBD-noDouble"

ak = pd.read_csv(dataset+"/actinic keratosis/metadata.csv", low_memory=False)
# Remove duplicate on lesion_id
ak_new = ak.drop_duplicates(subset=['lesion_id'], keep='first').copy()
# Remove if lesion_id is empty
#ak_new = ak_new.dropna(subset=['lesion_id'])
# Save new metadata file
ak_new.to_csv(dest+"/ActinicKeratosis/metadata.csv", index=False)
# Copy only unique images on lesion_id
print("Copying 'Actinic Keratosis' images...")
for index, row in ak_new.iterrows():
    copy_file(dataset+"/actinic keratosis/"+str(row.iloc[0])+".jpg", dest+"/ActinicKeratosis/"+str(row.iloc[0])+".jpg")

bcc = pd.read_csv(dataset+"/basal cell carcinoma/metadata.csv", low_memory=False)
bcc_new = bcc.drop_duplicates(subset=['lesion_id'], keep='first').copy()
#bcc_new = bcc_new.dropna(subset=['lesion_id'])
bcc_new.to_csv(dest+"/BasalCellCarcinoma/metadata.csv", index=False)
print("Copying 'Basal Cell Carcinoma' images...")
for index, row in bcc_new.iterrows():
    copy_file(dataset+"/basal cell carcinoma/"+str(row.iloc[0])+".jpg", dest+"/BasalCellCarcinoma/"+str(row.iloc[0])+".jpg")

mel = pd.read_csv(dataset+"/melanoma/metadata.csv", low_memory=False)
mel_new = mel.drop_duplicates(subset=['lesion_id'], keep='first').copy()
#mel_new = mel_new.dropna(subset=['lesion_id'])
mel_new.to_csv(dest+"/melanoma/metadata.csv", index=False)
print("Copying 'melanoma' images...")
for index, row in mel_new.iterrows():
    copy_file(dataset+"/melanoma/"+str(row.iloc[0])+".jpg", dest+"/melanoma/"+str(row.iloc[0])+".jpg")

nev = pd.read_csv(dataset+"/nevus/metadata.csv", low_memory=False)
nev_new = nev.drop_duplicates(subset=['lesion_id'], keep='first').copy()
#nev_new = nev_new.dropna(subset=['lesion_id'])
nev_new.to_csv(dest+"/nevus/metadata.csv", index=False)
print("Copying 'nevus' images...")
for index, row in nev_new.iterrows():
    copy_file(dataset+"/nevus/"+str(row.iloc[0])+".jpg", dest+"/nevus/"+str(row.iloc[0])+".jpg")

sk = pd.read_csv(dataset+"/seborrheic keratosis/metadata.csv", low_memory=False)
sk_new = sk.drop_duplicates(subset=['lesion_id'], keep='first').copy()
#sk_new = sk_new.dropna(subset=['lesion_id'])
sk_new.to_csv(dest+"/SeborrheicKeratosis/metadata.csv", index=False)
print("Copying 'Seborrheic Keratosis' images...")
for index, row in sk_new.iterrows():
    copy_file(dataset+"/seborrheic keratosis/"+str(row.iloc[0])+".jpg", dest+"/SeborrheicKeratosis/"+str(row.iloc[0])+".jpg")

scc = pd.read_csv(dataset+"/squamous cell carcinoma/metadata.csv", low_memory=False)
scc_new = scc.drop_duplicates(subset=['lesion_id'], keep='first').copy()
#scc_new = scc_new.dropna(subset=['lesion_id'])
scc_new.to_csv(dest+"/SquamousCellCarcinoma/metadata.csv", index=False)
print("Copying 'Squamous Cell Carcinoma' images...")
for index, row in scc_new.iterrows():
    copy_file(dataset+"/squamous cell carcinoma/"+str(row.iloc[0])+".jpg", dest+"/SquamousCellCarcinoma/"+str(row.iloc[0])+".jpg")

