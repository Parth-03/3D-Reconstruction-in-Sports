import os
import pickle

img_size = model.img_size
train_x_path = "image_train"
train_x = [] # images
train_y = [] # ground truth
i=0
with open("Panda.pkl", "rb") as file:
    data_list = pickle.load(file)
for sub_dir, _, files in os.walk(train_x_path):
    for filename in files:
        if i == 10:
            break

        corresponding_dict = None
        parent_folder = os.path.basename(sub_dir)
        image_identifier = os.path.join(parent_folder, filename)
        for data in data_list:
            if os.path.join(os.path.basename(os.path.dirname(data["img_path"])), os.path.basename(data["img_path"])) == image_identifier:
                corresponding_dict = data
                break
        if corresponding_dict == None:
            print("problem")
            print(image_identifier)
            continue
        file_path = os.path.join(sub_dir, filename)
        x, img_pil_nopad = open_image(file_path, img_size)
        train_x.append(x)
        y = corresponding_dict['params']
        train_y.append(np.array(y))
        print(i)
        i += 1

assert len(train_x) == len(train_y)# Size of dataset
