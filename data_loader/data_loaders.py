from torchvision import transforms
from base import BaseDataLoader
from utils.video_dataset import VideoFrameDataset, ImglistToTensor

SIZE = 160
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

class TrainDataLoader(BaseDataLoader):
    def __init__(self, root_path, annotationfile_path, batch_size, num_segments, frames_per_segment, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        trsfm = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize((SIZE,SIZE)),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
        
        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment

        self.dataset = VideoFrameDataset(
            root_path=self.root_path,
            annotationfile_path=self.annotationfile_path,
            num_segments=self.num_segments,
            frames_per_segment=self.frames_per_segment,
            imagefile_template='img_{:05d}.jpg',
            transform=trsfm,
            test_mode=False
        )

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class TestDataLoader(BaseDataLoader):
    def __init__(self, root_path, annotationfile_path, batch_size, num_segments, frames_per_segment, shuffle=True, validation_split=0.0, num_workers=1, training=False):

        trsfm = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize((SIZE,SIZE)),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
        
        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment

        self.dataset = VideoFrameDataset(
            root_path=self.root_path,
            annotationfile_path=annotationfile_path,
            num_segments=self.num_segments,
            frames_per_segment=self.frames_per_segment,
            imagefile_template='img_{:05d}.jpg',
            transform=trsfm,
            test_mode=False
        )

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
