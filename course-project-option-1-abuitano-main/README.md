# 14813/18813 Course Project Option 1
Author: Alonso Buitano - abuitano
Analysis of Soccer players statistics 2015-2022.

Please ensure Postgres is installed and running for task I
Postgres user name is 'postgres' by default, please also make sure you know the postgres password.

Pyspark is required for Tasks I, II and III. Version 3.3.0 was used locally, and 3.1.3 on GCP.

Tensorflow will be needed for Task III. Used version 2.9.2 locally and 2.5.0 on GCP.

Project_option_1.ipynb notebook has detailed implementation of the tasks with outputs.
Cloud_Project_option_1.ipynb notebook has the implementation on Google Cloud Platform, using a Dataproc cluster with Tensorflow 2.5.0 installed.
Colab_Project_option_1.ipynb notebook has the implementation on Google Colab.

Demo vide can be accessed [here](https://cmu.box.com/s/sgismjys49hhey6tvf1zbvq2jrrp7brq).

# Task I

The table schema was automatically inferred by pyspark when loading the data, with only minimal changes required during Task II feature engineering.

year and id columns were added during data ingestion to identify the year database the entries are from and give a unique serial id to the dataset.

The dataset contains soccer players' information from the FIFA game, from years 2015-2022. Features include name, club, various stats, contract info, etc.

### Feature descriptions and constraints:
sofifa_id integer, Player's id in sofifa.com website. Repeatable for same player different year.

player_url String, Player's url in sofifa.com website. Different for every entry.

short_name, long_name String, Player's name.

player_positions, club_position, nation_position Player's position, in club and national team.

overall integer (1-100), player overall score.

potential integer (1-100), player potential highest score.

value_eur integer, how much player is worth in euro.

wage_eur integer, player's salary in euro.

age integer, player age.

dob timestamp, player's date of birth.

height_cm integer, player height in cm.

weight_kg integer, player weight in kg.

club_name_id double, club id in database.

club_name String, name of the club player is in.

league_name String, league the club is in.

league_level integer, level the league is in.

club_jersey_number integer, nation_jersey_number String, player's jersey number in club and national teams.

club_loaned_from String, club that the player was loaned from.

club_joined timestamp, date that player joined the club.

club_contract_valid_until integer, year the player's contract with the club expires.

nationality_id integer, player's nation id.

nationality_name String, player's nationality.

nation_team_id double, the national team's id.

preferred_foot String, the player's preferred striking foot.

weak_foot integer (1-5).

skill_moves integer 1-5).

international_reputation integer (1-5), player's reputation.

work_rate String, the player's work rate (Low to High).

body_type String, player's body type.

real_face String (Boolean), whether the player's real face is used in the game.

release_clause_eur integer, amount in Euro for the player to be released before contract ends.

player_tags String, playstyle attribute tags given to the player.

player_traits String, player's traits.

pace, shooting, passing, dribbling, defending, physic integer (1-100), player's general stats.

attacking_crossing, attacking_finishing, attacking_heading_accuracy, attacking_short_passing, attacking_volleys integer, player's attacking-specific stats.

skill_dribbling, skill_curve, skill_fk_accuracy, skill_long_passing, skill_ball_control integer (1-100), player's ball handling skill stats.

movement_acceleration, movement_sprint_speed, movement_agility, movement_reactions, movement_balance integer (1-100), player's movement stats.

power_shot_power, power_jumping, power_stamina, power_strength, power_long_shots integer (1-100), player's power stats.

mentality_aggression, mentality_interceptions, mentality_positioning, mentality_positioning, mentality_vision, mentality_penalties, mentality_composure integer (1-100), player's mentality stats.

defending_marking_awareness, defending_standing_tackle, defending_sliding_tackle integer (1-100), player's defensive stats.

goalkeeping_diving, goalkeeping_handling, goalkeeping_kicking, goalkeeping_positioning, goalkeeping_reflexes, goalkeeping_speed integer (1-100), player's goalkeeping stats.

ls, st, rs, lw, lf, cf, rf, rw, lam, cam, ram, lm, lcm, cm, rcm, rm, lwb, ldm, cdm, rdm, rwb, lb, lcb, cb, rcb, rb, gk string, the player's base stat per position.

player_face_url String, url to the player's face in sofifa.com.

club_logo_url String, url to the club's logo in sofifa.com.

club_flag_url String, url to the club's flag in sofifa.com.

nation_logo_url String, url to the national team's logo in sofifa.com.

nation_flag_url String, url to the nation's team in sofifa.com.

year integer (2015-2022), year subset the entry belongs to.

id integer, non-repeatable. Row id.

# Task II

Here three functions are created to get the clubs with most players with contracts expiring in 2023, the clubs with the most average players aged over 27, across all years, and the most common national team positions for each year.

# Task III

For this task, the first step is doing data cleaning and feature engineering. Detailed data exploration is available on the Project_option_1.ipynb notebook.

The data types automatically inferred were mostly correct, with only a few changes needed. Mainly stats that were presented as a sum of two numbers were ingested as strings, so a conversion was needed to the correct values and data type.

Correlation analysis was carried out for all features, detecting and dropping columns with over 80% correlation to other features. There were quite a few columns with high correlation due to how similar the stats they represent are. For example, stats related to attacking all had high correlation, so only a few were kept and the rest were dropped.

Null values were counted for all columns and ones with over 50% nulls were dropped since they contain too little useful information.

I also looked at all the features to identify ones that don't provide any useful information to predict the player's overall. These columns were dropped as well (e.g. url columns).

### Pre-process pipeline

After doing data exploration to identify what is required for the data cleaning and feature engineering, I implemented a pipeline to carry out the stages and obtain a new dataframe containing only the vectorized, cleaned features and the outcome column which has the overall score for each player.

### Spark Machine Learning

The first machine learning models that were implemented are in Spark ML. I chose the Linear Regression model and the Random Forest regressor model.

Linear Regression model is a very standard regression model and it can compare to a similar implementation to a shallow Tensorflow linear network.
Random Forest regressor is a well performing model, and it's similar to the work we have done in the homeworks. I also initially tried a decision tree model, but Random Forest offers generally better performance so I went with that one.

The loss and metric used for all the ML models is Mean Squared Error (MSE). 

I implemented the models and trained without cross validation at first, then did the cross validation to compare the loss on the test dataset and see if there is any improvement after cross validation.

### Tensorflow

For the tensorflow implementation, since we can only create neural network models, the first one I implemented was a single layer with one output, This is effectively a simple linear regression model. The second model is a deep neural network with multiple layers and non-linear activation functions.

I chose adam optimizer for both the models.

For hyper-parameter tuning in the first model, since I didn't want to change the model structure I only tuned the learning rate.

The deeper model allowed me to chose the depth and width as hyperparameters for tuning.

All hyperparameter tuning was done using tensorboard. (Make sure to use the correct directories for tensorboard).

Note: the implementation of cross validation in the tensorflow section requires that the dataset is divisible exactly by the k number of folds, so that value may need adjusting due to the random split giving slightly varying dataset sizes.

# Task IV

The code was implemented to the cloud in Google Cloud Platform, using a Dataproc cluster with tensorflow version 2.5.0 installed during cluster initialization.
The notebook and data files were uploaded to the cloud.

I also implemented the code to Google Colab, uploading the relevant files to Google Drive.

For the cloud implementation, I removed the data exploration section of the code since it's not needed for any functionality and the pipeline carries out all the relevant preprocessing and feature engineering.
