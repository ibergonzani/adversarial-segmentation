
# Simple progress bar implemntations with customizable template
class ProgressBar():

	def __init__(self, total, prefix='', suffix='', use_percent=True, width=30,
					empty='-', full='#', template='{prefix} |{bar}| {ptot} {suffix}', show=False):
					
		assert (total > 0 and width > 0)
		
		self.total = total
		self.prefix = prefix
		self.suffix = suffix
		self.progress = 0
		self.use_percent = use_percent
		self.width = width
		self.empty = empty[:1]
		self.full = full[:1]
		self.template = template
		
		if show:
			self.update_and_show(progress=0)
	
	
	def _percent_string(self, percent):
		return '{:3d}%'.format(int(percent))
	
	
	def _build_progressbar(self):
		
		sit = self.progress / self.total
		n_full = int(self.width * sit)
		n_empty = self.width - n_full
		
		bar = self.full * n_full + self.empty * n_empty
		ptot = self._percent_string(100*sit) if self.use_percent else '{:d}/{:d}'.format(self.progress, self.total)
		
		kwargs = {'prefix': self.prefix, 'bar': bar, 'ptot': ptot, 'suffix': self.suffix}
		fullbar = self.template.format(**kwargs)
		
		return fullbar
		
	
	def update_and_show(self, progress=None, prefix=None, suffix=None):
			assert (progress == None or (progress >= 0 and progress <= self.total))
			
			self.progress = progress if progress != None else min(self.progress + 1, self.total)
			self.prefix = prefix if prefix != None else self.prefix
			self.suffix = suffix if suffix != None else self.suffix
			
			print('\r' + self._build_progressbar(), end='')