Label = {0: 'airplane',
         1: 'automobile',
         2: 'bird',
         3: 'cat',
         4: 'deer',
         5: 'dog',
         6: 'frog',
         7: 'horse',
         8: 'ship',
         9: 'truck',}

# TODO: Find a better way for the prompt selection, whether random select a prompt from multiple choices
refined_Label = {}
for k, v in Label.items():
    if len(v.split(",")) == 1:
        select = v
    else:
        select = v.split(",")[0]
    refined_Label.update({k: select})
