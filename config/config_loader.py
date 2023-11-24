import yaml

class ConfigLoader:
    registered_classes = {}

    def __init__(self, config_data=None, config_file=''):
        self.config_file = config_file
        self.config_data = config_data or self.load_config()
        self.objects = {}

    def load_config(self):
        if self.config_file:
            with open(self.config_file, "r") as file:
                config_data = yaml.safe_load(file)
        else:
            config_data = {}
        return config_data

    def load_objects(self):
        for name, obj_config in self.config_data.items():
            self.create_or_get_object(name, obj_config)

    def register_class(cls):
        ConfigLoader.registered_classes[cls.__name__] = cls
        return cls

    def __getitem__(self, name):
        return self.create_or_get_object(name, self.config_data[name])

    def _is_obj(self, obj_dict):
        if not isinstance(obj_dict, dict):
            return False
        return 'ref' in obj_dict or 'class_name' in obj_dict

    def create_or_get_object(self, name, obj_config):
        if name in self.objects:
            return self.objects[name]
        
        if obj_config.get("ref"):
            # use reference object from global config
            name = obj_config["ref"]
            if name not in self.config_data:
                raise ValueError(f"Reference object {name} not found in config")

            obj_config = self.config_data[obj_config["ref"]]
            return self.create_or_get_object(name, obj_config)

        # create object
        class_name = obj_config["class_name"]
        params = obj_config.get("params", {})

        for param_name, param_value in params.items():
            # if it is a dictionary, recursively create or get the object
            if isinstance(param_value, dict):
                param_value = self.create_or_get_object(name + "." + param_name, param_value)
            if isinstance(param_value, list) and self._is_obj(param_value[0]):
                param_value = [self.create_or_get_object(name + "." + param_name, item) for item in param_value]
            params[param_name] = param_value

        obj = ConfigLoader.registered_classes[class_name](**params)
        self.objects[name] = obj
        return obj