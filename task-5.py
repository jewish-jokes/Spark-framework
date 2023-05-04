import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, sum, dense_rank, when, lower
from pyspark.sql.window import Window


spark = SparkSession.builder \
    .appName("MyApp") \
    .master("local[*]") \
    .config("spark.driver.extraClassPath", "postgresql-42.6.0.jar") \
    .config("spark.jars", "postgresql-42.6.0.jar") \
    .getOrCreate()

jdbcUrl = "jdbc:postgresql://pagila:5432/postgres"
connectionProperties = {
  "user": "postgres",
  "password": "123456",
  "driver": "org.postgresql.Driver"
}

df_actor = spark.read.jdbc(url=jdbcUrl, table="actor", properties=connectionProperties)
df_address = spark.read.jdbc(url=jdbcUrl, table="address", properties=connectionProperties)
df_category = spark.read.jdbc(url=jdbcUrl, table="category", properties=connectionProperties)
df_country = spark.read.jdbc(url=jdbcUrl, table="country", properties=connectionProperties)
df_customer = spark.read.jdbc(url=jdbcUrl, table="customer", properties=connectionProperties)

df_film = spark.read.jdbc(url=jdbcUrl, table="film", properties=connectionProperties)
df_film_actor = spark.read.jdbc(url=jdbcUrl, table="film_actor", properties=connectionProperties)
df_film_category = spark.read.jdbc(url=jdbcUrl, table="film_category", properties=connectionProperties)
df_inventory = spark.read.jdbc(url=jdbcUrl, table="inventory", properties=connectionProperties)
df_language = spark.read.jdbc(url=jdbcUrl, table="language", properties=connectionProperties)

df_payment = spark.read.jdbc(url=jdbcUrl, table="payment", properties=connectionProperties)
df_rental = spark.read.jdbc(url=jdbcUrl, table="rental", properties=connectionProperties)
df_staff = spark.read.jdbc(url=jdbcUrl, table="staff", properties=connectionProperties)
df_store = spark.read.jdbc(url=jdbcUrl, table="store", properties=connectionProperties)
df_city = spark.read.jdbc(url=jdbcUrl, table="city", properties=connectionProperties)

# task 1
df_category.join(df_film_category,"category_id") \
    .select(df_category["name"].alias("category_name"), df_film_category["film_id"]) \
    .groupBy("category_name") \
    .agg({"film_id": "count"}) \
    .withColumnRenamed("count(film_id)", "films_count") \
    .orderBy(col("films_count").desc()).show()

# task 2
df_actor.join(df_film_actor, "actor_id") \
    .join(df_inventory, "film_id") \
    .join(df_rental, "inventory_id") \
    .groupBy("first_name", "last_name") \
    .agg(count("rental_id").alias("rental_amount")) \
    .sort(desc("rental_amount")) \
    .limit(5) \
    .show()

# task 3
df_category.join(df_film_category, "category_id", "inner") \
    .join(df_inventory, "film_id") \
    .join(df_rental, "inventory_id") \
    .join(df_payment, "rental_id") \
    .select(col("name").alias("category_name"), col("amount")) \
    .groupby("category_name") \
    .agg(sum("amount").alias("payment_amount")) \
    .orderBy(col("payment_amount").desc()) \
    .limit(1) \
    .show()

# task 4
df_film.join(df_inventory, "film_id", "left") \
    .filter(col("inventory_id").isNull()) \
    .select("title") \
    .show()

# task 5
category_id = df_category.filter(col("name") == "Children") \
    .select("category_id") \
    .limit(1)

df_actor_children_movies = (
df_film_category.join(category_id, "category_id") \
    .join(df_film_actor, "film_id") \
    .join(df_actor, "actor_id") \
    .groupBy("actor_id", "first_name", "last_name") \
    .agg(count("*").alias("movie_count"))
)

window_spec = Window.orderBy(col("movie_count").desc())

df_top_actors = (
df_actor_children_movies.withColumn("rank", dense_rank().over(window_spec)) \
    .filter(col("rank") <= 3) \
    .select("first_name", "last_name", "movie_count") \
    .orderBy(col("movie_count").desc())
)

df_top_actors.show()

# task 6
df_active_customers = df_city.join(df_address, "city_id") \
    .join(df_customer, "address_id") \
    .groupBy(df_city["city"]) \
    .agg(count(when(df_customer.active == 1, 1)).alias('active_customers'), count(when(df_customer.active == 0, 1)).alias('inactive_customers')) \
    .orderBy('inactive_customers', ascending=False)

df_active_customers.show()

# task 7

joined_data = df_city \
    .join(df_address, "city_id") \
    .join(df_customer, "address_id") \
    .join(df_rental, "customer_id") \
    .join(df_inventory, "inventory_id") \
    .join(df_film, "film_id") \
    .join(df_film_category, "film_id") \
    .join(df_category, "category_id") 

filtered_data = joined_data \
    .filter(lower(df_film["title"]).like("a%")) \
    .filter(lower(df_city["city"]).like("%-%"))

grouped_data = filtered_data \
    .groupBy(df_category["name"]) \
    .agg(sum((df_rental["return_date"] - df_rental["rental_date"])).alias("time_diff"))

window_spec = Window.orderBy(grouped_data["time_diff"].desc())

ranked_data = grouped_data.withColumn("rank_sum", dense_rank().over(window_spec))

final_data = ranked_data \
    .filter(ranked_data["rank_sum"] <= 1) \
    .select(df_category["name"].alias("category_name"))

final_data.show()
