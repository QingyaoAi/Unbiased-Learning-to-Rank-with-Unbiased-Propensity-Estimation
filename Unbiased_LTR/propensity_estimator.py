import os,sys
import json, copy, random
import click_models as CM
import data_utils


class RandomizedPropensityEstimator:
	def __init__(self, file_name=None):
		# If file_name is not None, 
		if file_name:
			self.loadEstimatorFromFile(file_name)

	def getPropensityForOneList(self, click_list, use_non_clicked_data=False):
		propensity_weights = []
		for r in xrange(len(click_list)):
			pw = 0.0
			if use_non_clicked_data or (click_list[r] > 0):
				pw = self.IPW_list[r]
			propensity_weights.append(pw)
		return propensity_weights

	def loadEstimatorFromFile(self, file_name):
		with open(file_name) as data_file:	
			data = json.load(data_file)
			self.click_model = CM.loadModelFromJson(data['click_model'])
			self.IPW_list = data['IPW_list']
		return

	def estimateParametersFromModel(self, click_model, training_data):
		self.click_model = click_model
		click_count = [[0 for _ in xrange(x+1)] for x in xrange(training_data.rank_list_size)]
		label_lists = copy.deepcopy(training_data.gold_weights)
		# Conduct randomized click experiments
		session_num = 0
		while session_num < 10e6:
			index = random.randint(0,len(label_lists)-1)
			random.shuffle(label_lists[index])
			click_list, _, _ = self.click_model.sampleClicksForOneList(label_lists[index])
			# Count how many clicks happened on the i position for a list with that lengths.
			for i in xrange(len(click_list)):
				click_count[len(click_list)-1][i] += click_list[i]
			session_num += 1
		# Count how many clicks happened on the 1st position for a list with different lengths.
		first_click_count = [0 for _ in xrange(training_data.rank_list_size)]
		agg_click_count = [0 for _ in xrange(training_data.rank_list_size)]
		for x in xrange(len(click_count)):
			for y in xrange(x,len(click_count)):
				first_click_count[x] += click_count[y][0]
				agg_click_count[x] += click_count[y][x]

		# Estimate IPW for each position (assuming that position 0 has weight 1)
		self.IPW_list = [min(first_click_count[x]/(agg_click_count[x]+10e-6), first_click_count[x]) for x in xrange(len(click_count))]
		return

	def outputEstimatorToFile(self, file_name):
		json_dict = {
			'click_model' : self.click_model.getModelJson(),
			'IPW_list' : self.IPW_list
		}
		with open(file_name, 'w') as fout:
			fout.write(json.dumps(json_dict, indent=4, sort_keys=True))
		return


class OraclePropensityEstimator:

	def __init__(self, click_model):
		self.click_model = click_model

	def getPropensityForOneList(self, click_list, use_non_clicked_data=False):
		return self.click_model.estimatePropensityWeightsForOneList(click_list, use_non_clicked_data)



def main():
	click_model_json_file = sys.argv[1]
	data_dir = sys.argv[2]
	output_file = sys.argv[3]
	
	print("Load data from " + data_dir)
	train_set = data_utils.read_data(data_dir, 'train')
	click_model = None
	with open(click_model_json_file) as fin:
		model_desc = json.load(fin)
		click_model = CM.loadModelFromJson(model_desc)
	print("Estimating...")
	estimator = RandomizedPropensityEstimator()
	estimator.estimateParametersFromModel(click_model, train_set)
	print("Output results...")
	estimator.outputEstimatorToFile(output_file)


if __name__ == "__main__":
	main()
