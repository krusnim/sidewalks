

class CategoricalBlameModel:

    def __init__(self, issues, complaints, radius=200):
        self.issue_weights = {
            "CONC_DEFECTS": 0,  # Concrete defects
            "STORMGRATES": 0,   # Storm grate placement
            "SW_ATTRIBUTE": 0,  # Sidewalk atributes like width, material changes, etc. Sort of a "misc".
            "DRIVEWAY": 0,      # Driveways intersecting the sidewalk
            "POLE": 0,          # Poles in the path of travel
            "SIGN": 0,          # Signs in the path of travel
            "SW_CS": 0,         # Sidewalk cross-slope (twistedness)
            "BOXES": 0,         # Above-grade boxes / overhangs
            "CASTINGS": 0       # Unclear exactly what this means - seems debris related
        }

        type_counts = issues.groupby("EVNT_TYPE").size()

        compl = complaints.buffer(radius)

        for i in range(len(compl)):
            area = compl.iloc[i]
            near_issues = issues[~ issues.geometry.intersection(area).is_empty]
            print(f"{len(near_issues)} issues near complaint")
            for j in range(len(near_issues)):
                ty = near_issues.iloc[j]["EVNT_TYPE"]
                self.issue_weights[ty] += 1 / len(near_issues) / type_counts[ty]
    
    def blame(self, issue):
        ty = issue["EVNT_TYPE"]
        if not (ty in self.issue_weights.keys()):
            # print(f"Warning: nonstandard event type {ty}")
            return 0
        return self.issue_weights[ty]

    def fairness_blame(self, p, history):
        d = sum([p.distance(q) for q in history])
        return d

