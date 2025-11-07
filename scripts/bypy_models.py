import bypy

# 封装百度云操作类
class ByPy:
	# 初始化
	def __init__(self):
		self.bp = bypy.ByPy()
	
	# 上传文件
	def upload(self, local_path, remote_path):
		self.bp.upload(local_path, remote_path)
	
	# 下载文件
	def download(self, remote_path, local_path):
		self.bp.download(remote_path, local_path)

    # 删除文件
	def delete(self, remote_path):
		self.bp.delete(remote_path)

    # 列出文件
	def list(self, remote_path):
		self.bp.list(remote_path)

    # 创建文件夹
	def mkdir(self, remote_path):
		self.bp.mkdir(remote_path)

    # 删除文件夹
	def rmdir(self, remote_path):
		self.bp.rmdir(remote_path)

    # 移动文件
	def move(self, remote_path, new_remote_path):
		self.bp.move(remote_path, new_remote_path)

	# 复制文件
	def copy(self, remote_path, new_remote_path):
		self.bp.copy(remote_path, new_remote_path)

	# 重命名文件
	def rename(self, remote_path, new_remote_path):
		self.bp.rename(remote_path, new_remote_path)

	# 获取文件信息
	def info(self, remote_path):
		self.bp.info(remote_path)

	# 同步syncup 
	def sync(self, local_path, remote_path):
		self.bp.syncup(local_path, remote_path)


if __name__ == '__main__':
	bypy = ByPy()
	bypy.download('/dash/mysql', 'E:/Project/work/llm-agent-app/docs/bypy')
	