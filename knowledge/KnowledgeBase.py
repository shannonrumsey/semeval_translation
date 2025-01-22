import pandas as pd
class KnowledgeBase():
    def __init__(self, ent:pd.DataFrame, alias:pd.DataFrame):
        self.ent:pd.DataFrame = ent
        self.alias:pd.DataFrame = alias

    def get_name_from_alias(self, alias):
        return self.alias[self.alias['alias'] == alias]

    def get_translation_from_name(self, name, translation):
        return self.ent[self.ent['name'] == name][translation]

    def get(self, query, translation):
        name = self.get_name_from_alias(query)

        if len(name) == 0:
            return self.get_translation_from_name(query, translation)
        else:
            return self.get_translation_from_name(name[0], translation)
