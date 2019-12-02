import pymysql as mysql
import base64
import datetime
import time

class Mysql():
	def __init__(self, host="192.168.0.175", port=3306):
		self.host = host
		self.port = port
		self.user = "titi"
		self.passwd = "titi861203"
		self.db = "mydb"

		self.sqlTime = None

	def getImage(self):
		while True:
			try:
				db = mysql.connect(
					host = self.host,
					port = self.port,
					user = self.user,
					passwd = self.passwd,
					db = self.db
				)
				cursor = db.cursor()
				SQL = "SELECT * FROM ros_image ORDER BY time DESC LIMIT 1"
				cursor.execute(SQL)
				results = cursor.fetchone()
				now_time = datetime.datetime.now()
				sql_time = results[0]
				print(sql_time)
				if self.is_it_now(now_time, sql_time):
					self.sqlTime = sql_time
					fout = open("image.jpg", "wb")
					fout.write(results[1])
					fout.close()
					cursor.close()
					db.close()
					break
				cursor.close()
				db.close()
			except:
				print("MySQL connect error!")
				break
			time.sleep(3)

	def is_it_now(self, now_time, sql_time):
		now_seconds = time.mktime(now_time.timetuple())
		sql_seconds = time.mktime(sql_time.timetuple())
		delta = now_seconds - sql_seconds
		if delta <= 5:  return True
		else:   return False

	def updateImage(self, X, Y):
		print("[updateImage]", self.sqlTime)
		try:
			db = mysql.connect(
				host = self.host,
				port = self.port,
				user = self.user,
				passwd = self.passwd,
				db = self.db
			)
			cursor = db.cursor()
			localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
			SQL = "INSERT INTO ros_results (time, middleX, middleY) VALUES ('"+localtime+"', '%s', '%s')" %(X, Y)
			cursor.execute(SQL)
			db.commit()
			cursor.close()
			db.close()
		except:
			print("MySQL update error!")
 

