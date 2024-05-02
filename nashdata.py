import pandas
import geopandas
from shapely.geometry import Polygon, Point

import blame

class RawData:
    def __init__(self, data_folder):
        print("Loading complaints...")
        self.df_complaints_raw = pandas.read_csv(f"{data_folder}/complaints.csv")
        print("Loading sidewalks...")
        self.df_sidewalks_raw = pandas.read_csv(f"{data_folder}/sidewalks.csv")
        print("Loading ZIP codes...")
        self.df_zip_raw = geopandas.read_file(f"{data_folder}/zipcode_polygons.gdb")

class SidewalkData:

    def __init__(self, raw):
        self.df_complaints_raw = raw.df_complaints_raw
        self.df_sidewalks_raw  = raw.df_sidewalks_raw
        self.df_zip_raw        = raw.df_zip_raw

        self.hist_len = 300
        self.radius = 100
        self._filter_data()
        
    def _filter_data(self):

        # =====
        # ZIP CODE DATA
        # This is mostly for visualization purposes

        zip_codes = {
            37218: "Bordeaux",
            37207: "North Nashville",
            37115: "Madison",
            37138: "Old Hickory",
            37216: "Inglewood",
            37209: "Sylvan Park",
            37208: "Germantown",
            37203: "Gulch / West End",
            37201: "Downtown / Sobro",
            37210: "South Nashville",
            37206: "East Nashville",
            37212: "Music Row / Vanderbilt",
            37214: "Donelson",
            37076: "Hermitage",
            37205: "Belle Meade",
            37215: "Green Hills",
            37204: "Belmont / 12 South",
            37220: "Oak Hill",
            37211: "Nolensville Pike",
            37217: "Briley Parkway / Percy Priest"
        }
        nash_zips = [k for k in zip_codes.keys()]

        df_zip_names = pandas.DataFrame.from_dict(zip_codes, orient="index")
        df_zip = self.df_zip_raw
        df_zip = df_zip.set_crs("EPSG:4326")
        df_zip["ZIP_CODE"] = df_zip["ZIP_CODE"].astype(int)
        df_zip = df_zip[df_zip["ZIP_CODE"].isin(nash_zips)]
        df_zip = df_zip.to_crs("ESRI:103526")


        # =====
        # COMPLAINT DATA

        valid_complaints = [
            # Some of these are kind of misleading (and noisy)
            #   so I'll try to motivate my choices here.
            
            # EXPLICIT: These should be obvious.
            "Sidewalk", 
            "Modify Sidewalk", 
            "Damaged Sidewalk", 
            "Info or Status of Sidewalk",
            "Request New Sidewalk", 

            # FEATURES: These refer to road features intersecting the sidewalk.
            #   They are covered by the ADA standards and therefore the sidewalk data.
            "New Curb Request", 
            "New Rail", 

            # PAVING: Paving requests include both sidewalk paving and road paving.
            #   This is less annoying that it seems: it'd be unusual to request a paved
            #   sidewalk without an existing paved road in a city like Nashville,
            #   but there are lots of unpaved sidewalks by roads, some represented in the 
            #   sidewalk issues database.
            #   So, when we get a paving request near a sidewalk issue, chances are that:
            #   (1) there is a sidewalk to make an issue about,
            #   (2) there is already a street paved there (since there's a sidewalk)
            #   (3) the sidewalk is unpaved, since the complaint couldn't be about the
            #       (already paved) street. 
            "Paving Request", 
            "Info on Paving",
            "Request Road to be Paved", 
            "Request New", 

            # BIKES: Bike access is included as an accessibility standard alongside
            #   the sidewalk data - areas impassable by bike are noted. Annoyingly,
            #   though, this is not generally distinguishable from issues about
            #   the sidewalks because the  
            #   
            "Request for a New / Improved Bikeway", 
            "Request a New/Improved Bikeway"

            # Finally, note that there's not much harm in including complaints where there are
            # no sidewalks, because we only care about complaints that are proximal to issues,
            # and no sidewalks => no reported issues. So when we include e.g. "New Sidewalk", 
            # many of them fall away from any issues, but a few of them blame issues
            # like SW_ATTRIBUTE - which checks for unpaved paths, among other things.
            ]

        # Open and format data (incl. geodata)
        df_complaints = self.df_complaints_raw
        print(f"Total complaints: {len(df_complaints)}")
        df_complaints = df_complaints[df_complaints["ZIP"].notnull()]
        print(f"\tWith ZIP: {len(df_complaints)}")
        df_complaints = df_complaints[df_complaints["ZIP"].isin(zip_codes.keys())]
        print(f"\tIn valid ZIP: {len(df_complaints)}")
        df_complaints = df_complaints.rename(columns={"ZIP": "ZIP_CODE"})
        df_complaints = geopandas.GeoDataFrame(
            df_complaints, geometry=geopandas.points_from_xy(df_complaints["Longitude"], df_complaints["Latitude"])
        )
        df_complaints = df_complaints.set_crs("EPSG:4326")
        df_complaints = df_complaints.to_crs("ESRI:103526")
        df_complaints["Date / Time Opened"] = pandas.to_datetime(df_complaints["Date / Time Opened"], format="%m/%d/%Y %I:%M:%S %p")

        # Scope to relevant complaints only
        df_complaints = df_complaints[df_complaints["Request Type"]=="Streets, Roads & Sidewalks"]
        print(f"\tWith roads request: {len(df_complaints)}")
        df_complaints = df_complaints[df_complaints["Subrequest Type"].isin(valid_complaints)]
        print(f"\tWith sidewalk subrequest: {len(df_complaints)}")

        # Scope to complaints that occurred after the survey
        survey_date = pandas.to_datetime("05/29/2019 12:30:00 AM")
        df_complaints = df_complaints[df_complaints["Date / Time Opened"] > survey_date]
        print(f"\tNewer than survey {len(df_complaints)}")

        # Subframe: My own ZIP code ("presumed-available" data)


        # =====
        # SIDEWALK DATA

        # Filter sidewalk info and add geodata
        df_sidewalks = geopandas.GeoDataFrame(
            self.df_sidewalks_raw, geometry=geopandas.points_from_xy(self.df_sidewalks_raw["EVNT_LON"], self.df_sidewalks_raw["EVNT_LAT"])
        )
        df_sidewalks = df_sidewalks.set_crs("EPSG:4326")
        df_sidewalks = df_sidewalks.sjoin(self.df_zip_raw, how="left")
        df_sidewalks["ZIP_CODE"] = df_sidewalks["ZIP_CODE"].astype(int)
        print(f"Total issues: {len(df_sidewalks)}")
        df_sidewalks = df_sidewalks[df_sidewalks["ZIP_CODE"].isin(nash_zips)]
        print(f"\tIn valid ZIP: {len(df_sidewalks)}")
        df_sidewalks = df_sidewalks.to_crs("ESRI:103526")
        self.city = df_sidewalks[~df_sidewalks["REPAIRED"].isna()]
        df_sidewalks = df_sidewalks[df_sidewalks["REPAIRED"].isna()]
        print(f"\tNot repaired: {len(df_sidewalks)}")


        # =====

        train_complaints = df_complaints[df_complaints["ZIP_CODE"].astype(int) == 37212]
        print(f"Training with {len(train_complaints)} complaints")
        train_issues = df_sidewalks[df_sidewalks["ZIP_CODE"] == 37212]
        print(f"\tand {len(train_issues)} issues")
        self.blame_model = blame.CategoricalBlameModel(
            train_issues,
            train_complaints,
            radius=self.radius
        )

        self.df_sidewalks = df_sidewalks
        self.df_complaints = df_complaints
        self.df_zip = df_zip

        self.area = Polygon(df_zip.unary_union.exterior.coords)
        self.survey_hist = []

    def clear_hist(self):
        self.survey_hist = []

    def score(self, x, y, include_repaired=False, fair_weight=0):
        point = Point(x, y)
        area = point.buffer(self.radius)

        # These two lines are very slow but probably unavoidable
        point_intersection = self.df_sidewalks.intersection(area)
        nearby_issues = self.df_sidewalks[~ point_intersection.is_empty]

        if not include_repaired:
            nearby_issues = nearby_issues[nearby_issues["REPAIRED"]!=1]

        res = 0
        for i in range(len(nearby_issues)):
            issue = nearby_issues.iloc[i]
            res += self.blame_model.blame(issue)

        if fair_weight > 0:
            p = Point(x, y)
            res += fair_weight * self.blame_model.fairness_blame(p, self.survey_hist)
        
        self.update_survey_history(x, y)
        return res

    def is_valid_survey(self, x, y):
        p = Point(x, y)
        for q in self.survey_hist:
            if p.distance(q) < self.radius:
                return False
        return True

    def update_survey_history(self, x, y):
        p = Point(x, y)
        self.survey_hist += [p]
        n = min(self.hist_len, len(self.survey_hist)+1)
        self.survey_hist = self.survey_hist[-n:]

