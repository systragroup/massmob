import polars as pl

class IntegrityModel:

    def test_phone_ids_integrity(self):
        """
        Checks that all phone_id values present in points are referenced in the phone_id column itself (self-integrity).
        Returns True if OK, False otherwise.
        """
        # Get all unique phone_ids in the dataframe
        phone_ids = self.points['phone_id'].unique()
        # Filter rows where phone_id is in the list of unique phone_ids
        filtered_points = self.points.filter(
            pl.col('phone_id').is_in(phone_ids)
        )
        # Compare the number of original rows and filtered rows
        return len(self.points) == len(filtered_points)
    

    def integrity_test_all(self):
        """
        Runs all integrity test methods (starting with 'test_') and returns a dictionary of results.
        """
        results = {}
        # Loop through all attributes, find methods that start with 'test_'
        for attr in dir(self):
            if attr.startswith('test_') and callable(getattr(self, attr)):
                method = getattr(self, attr)
                try:
                    results[attr] = method()
                except Exception as e:
                    results[attr] = f"Error: {e}"
        return results
