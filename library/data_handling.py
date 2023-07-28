"""
handling of data provided by chaimeleon 
"""

import os
import json


class ChaimeleonData:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def process_data_directory(self):
        pass

    def add_case_data(self, case_directory):
        pass

    def prepare_image_file(self, image_file):
        pass


class ProstateCancerDataset(ChaimeleonData):
    def __init__(self, data_directory):
        super().__init__(data_directory)

    def parse_metadata(self, metadata_file):
        pass

    def parse_ground_truth(self, ground_truth_file):
        pass


class LungCancerDataset(ChaimeleonData):
    def __init__(self, data_directory):
        super().__init__(data_directory)

    def parse_metadata(self, metadata_file):
        pass

    def parse_ground_truth(self, ground_truth_file):
        pass
