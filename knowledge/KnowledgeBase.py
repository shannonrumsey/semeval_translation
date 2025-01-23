import pandas as pd
import os

path = lambda x: os.path.join(os.path.dirname(__file__), x)


class KnowledgeBase():
    """Knowledge Base Class

        `get_name_from_alias(str)` -> returns `str` or `None`

        `get_translation_from_name(str, str)` -> returns `str` or `None`

        `get(str, str)` -> returns `str` or `None`

        ** Example Usage **
        ```
        from knowledge import KnowledgeBase

        kb = KnowledgeBase()
        spanish_usa = kb.get("U.S.A", "es") -> "Estados Unidos"

        alias_usa = kb.get_name_from_alias("U.S.A") -> "United States"
        ```
        I would recommend just using get() for any query as the get command will utilize both get_name_from_alias, and get_translation_from_name
    """
    def __init__(self):
        ent = pd.read_csv(path('ent_kb.csv'))
        alias = pd.read_csv(path('alias_kb.csv'))

        self.ent:pd.DataFrame = ent
        self.alias:pd.DataFrame = alias

    def get_name_from_alias(self, alias:str) -> str:
        """Get a name of an entity from its alias
        
        Args:
            alias: str, The alias we want to search for 
        Returns:
            If found will return the string name of the entity 
            Other wise will return None
        """
        try:
            name = self.alias[self.alias['alias'] == alias]['name']
            if not name.empty:
                return name.iloc[0]
            else:
                return None
        except:
            return None

    def get_translation_from_name(self, name:str, translation:str) -> str:
        """Get a translation of an entity given its name and desired translation
        
        Args:
            name: str, The name we want the translation for
            translation: str, from ['ar', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'th', 'tr', 'zh']
        Returns:
            If found will return the translation
            Else will return None
        """
        try:
            translate = self.ent[self.ent['name'] == name][translation]
            if not translate.empty:
                return translate.iloc[0]
            else:
                return None
        except:
            return None
    
    def get_entity_from_id(self, identification:str) -> str:
        """Get an entity given its id
        
        Args:
            identification: str, The id we want the entity for
        Returns:
            If found will return the entity
            Else will return None
        """
        try:
            name = self.ent[self.ent['id'] == identification]['name']
            if not name.empty:
                return name.iloc[0]
            else:
                return None
        except:
            return None
        
    def get(self, query:str, translation:str):
        """ Wrapper function that will automatically look for the name in the alias table and then return the
        found translaton from the ent table 

        Args:
            `query: str`, The search query (either an alias or a name)
            `translation: str`, The desired translation from this list: `['ar', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'th', 'tr', 'zh']`
        Returns:
            If found returns translation as str,
            Else returns None
        """

        name = self.get_name_from_alias(query) #attempt to get the name from alias

        if name is None: #if not an alias treat as the actual name
            try:
                return self.get_translation_from_name(query, translation)
            except:
                pass
        else: #if an alias use name from get_alias command to get translation
            try:
                return self.get_translation_from_name(name, translation)
            except:
                pass
        
        return None #nothing found return None
