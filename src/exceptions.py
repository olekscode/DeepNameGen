class DriveSecretsNotFound(Exception):
    def __init__(self):
        super(DriveSecretsNotFound, self).__init__(
            'In order to use DriveLogger you must get client_secrets.json ' +
            'and put it inside src/model/ folder. Visit https://github.com/ObjectProfile/PredictingMethodNames ' +
            'for detailed instructions.')
