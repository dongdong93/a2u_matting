import torch


PATH = '/PATH_TO/model.pth'
SAVE_PATH = '/PATH_TO/model_save.pth'
checkpoint = torch.load(PATH, map_location='cpu')
state = {
    'state_dict': checkpoint['state_dict']
}
torch.save(state, SAVE_PATH)
