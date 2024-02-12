# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HuggingFace datasets implementation of the json tasks in the BIG-Bench Dataset.
For the programatic tasks, please use the BIG-Bench API on github.com/google/BIG-bench.
"""


from typing import Optional

import bigbench.api.util as bb_utils  # From: "bigbench @ https://storage.googleapis.com/public_research_data/bigbench/bigbench-0.0.1.tar.gz"
import bigbench.bbseqio.bigbench_bridge as bbb
from bigbench.api import json_task
from bigbench.bbseqio import bigbench_json_paths as bb_json_paths
from sentencepiece import sentencepiece_model_pb2  # noqa: this is also required by bigbench.api.util

import datasets


logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2206.04615,
  doi = {10.48550/ARXIV.2206.04615},
  url = {https://arxiv.org/abs/2206.04615},
  author = {Srivastava, Aarohi and Rastogi, Abhinav and Rao, Abhishek and Shoeb, Abu Awal Md and Abid, Abubakar and Fisch, Adam and Brown, Adam R. and Santoro, Adam and Gupta, Aditya and Garriga-Alonso, Adrià and Kluska, Agnieszka and Lewkowycz, Aitor and Agarwal, Akshat and Power, Alethea and Ray, Alex and Warstadt, Alex and Kocurek, Alexander W. and Safaya, Ali and Tazarv, Ali and Xiang, Alice and Parrish, Alicia and Nie, Allen and Hussain, Aman and Askell, Amanda and Dsouza, Amanda and Slone, Ambrose and Rahane, Ameet and Iyer, Anantharaman S. and Andreassen, Anders and Madotto, Andrea and Santilli, Andrea and Stuhlmüller, Andreas and Dai, Andrew and La, Andrew and Lampinen, Andrew and Zou, Andy and Jiang, Angela and Chen, Angelica and Vuong, Anh and Gupta, Animesh and Gottardi, Anna and Norelli, Antonio and Venkatesh, Anu and Gholamidavoodi, Arash and Tabassum, Arfa and Menezes, Arul and Kirubarajan, Arun and Mullokandov, Asher and Sabharwal, Ashish and Herrick, Austin and Efrat, Avia and Erdem, Aykut and Karakaş, Ayla and Roberts, B. Ryan and Loe, Bao Sheng and Zoph, Barret and Bojanowski, Bartłomiej and Özyurt, Batuhan and Hedayatnia, Behnam and Neyshabur, Behnam and Inden, Benjamin and Stein, Benno and Ekmekci, Berk and Lin, Bill Yuchen and Howald, Blake and Diao, Cameron and Dour, Cameron and Stinson, Catherine and Argueta, Cedrick and Ramírez, César Ferri and Singh, Chandan and Rathkopf, Charles and Meng, Chenlin and Baral, Chitta and Wu, Chiyu and Callison-Burch, Chris and Waites, Chris and Voigt, Christian and Manning, Christopher D. and Potts, Christopher and Ramirez, Cindy and Rivera, Clara E. and Siro, Clemencia and Raffel, Colin and Ashcraft, Courtney and Garbacea, Cristina and Sileo, Damien and Garrette, Dan and Hendrycks, Dan and Kilman, Dan and Roth, Dan and Freeman, Daniel and Khashabi, Daniel and Levy, Daniel and González, Daniel Moseguí and Perszyk, Danielle and Hernandez, Danny and Chen, Danqi and Ippolito, Daphne and Gilboa, Dar and Dohan, David and Drakard, David and Jurgens, David and Datta, Debajyoti and Ganguli, Deep and Emelin, Denis and Kleyko, Denis and Yuret, Deniz and Chen, Derek and Tam, Derek and Hupkes, Dieuwke and Misra, Diganta and Buzan, Dilyar and Mollo, Dimitri Coelho and Yang, Diyi and Lee, Dong-Ho and Shutova, Ekaterina and Cubuk, Ekin Dogus and Segal, Elad and Hagerman, Eleanor and Barnes, Elizabeth and Donoway, Elizabeth and Pavlick, Ellie and Rodola, Emanuele and Lam, Emma and Chu, Eric and Tang, Eric and Erdem, Erkut and Chang, Ernie and Chi, Ethan A. and Dyer, Ethan and Jerzak, Ethan and Kim, Ethan and Manyasi, Eunice Engefu and Zheltonozhskii, Evgenii and Xia, Fanyue and Siar, Fatemeh and Martínez-Plumed, Fernando and Happé, Francesca and Chollet, Francois and Rong, Frieda and Mishra, Gaurav and Winata, Genta Indra and de Melo, Gerard and Kruszewski, Germán and Parascandolo, Giambattista and Mariani, Giorgio and Wang, Gloria and Jaimovitch-López, Gonzalo and Betz, Gregor and Gur-Ari, Guy and Galijasevic, Hana and Kim, Hannah and Rashkin, Hannah and Hajishirzi, Hannaneh and Mehta, Harsh and Bogar, Hayden and Shevlin, Henry and Schütze, Hinrich and Yakura, Hiromu and Zhang, Hongming and Wong, Hugh Mee and Ng, Ian and Noble, Isaac and Jumelet, Jaap and Geissinger, Jack and Kernion, Jackson and Hilton, Jacob and Lee, Jaehoon and Fisac, Jaime Fernández and Simon, James B. and Koppel, James and Zheng, James and Zou, James and Kocoń, Jan and Thompson, Jana and Kaplan, Jared and Radom, Jarema and Sohl-Dickstein, Jascha and Phang, Jason and Wei, Jason and Yosinski, Jason and Novikova, Jekaterina and Bosscher, Jelle and Marsh, Jennifer and Kim, Jeremy and Taal, Jeroen and Engel, Jesse and Alabi, Jesujoba and Xu, Jiacheng and Song, Jiaming and Tang, Jillian and Waweru, Joan and Burden, John and Miller, John and Balis, John U. and Berant, Jonathan and Frohberg, Jörg and Rozen, Jos and Hernandez-Orallo, Jose and Boudeman, Joseph and Jones, Joseph and Tenenbaum, Joshua B. and Rule, Joshua S. and Chua, Joyce and Kanclerz, Kamil and Livescu, Karen and Krauth, Karl and Gopalakrishnan, Karthik and Ignatyeva, Katerina and Markert, Katja and Dhole, Kaustubh D. and Gimpel, Kevin and Omondi, Kevin and Mathewson, Kory and Chiafullo, Kristen and Shkaruta, Ksenia and Shridhar, Kumar and McDonell, Kyle and Richardson, Kyle and Reynolds, Laria and Gao, Leo and Zhang, Li and Dugan, Liam and Qin, Lianhui and Contreras-Ochando, Lidia and Morency, Louis-Philippe and Moschella, Luca and Lam, Lucas and Noble, Lucy and Schmidt, Ludwig and He, Luheng and Colón, Luis Oliveros and Metz, Luke and Şenel, Lütfi Kerem and Bosma, Maarten and Sap, Maarten and ter Hoeve, Maartje and Farooqi, Maheen and Faruqui, Manaal and Mazeika, Mantas and Baturan, Marco and Marelli, Marco and Maru, Marco and Quintana, Maria Jose Ramírez and Tolkiehn, Marie and Giulianelli, Mario and Lewis, Martha and Potthast, Martin and Leavitt, Matthew L. and Hagen, Matthias and Schubert, Mátyás and Baitemirova, Medina Orduna and Arnaud, Melody and McElrath, Melvin and Yee, Michael A. and Cohen, Michael and Gu, Michael and Ivanitskiy, Michael and Starritt, Michael and Strube, Michael and Swędrowski, Michał and Bevilacqua, Michele and Yasunaga, Michihiro and Kale, Mihir and Cain, Mike and Xu, Mimee and Suzgun, Mirac and Tiwari, Mo and Bansal, Mohit and Aminnaseri, Moin and Geva, Mor and Gheini, Mozhdeh and T, Mukund Varma and Peng, Nanyun and Chi, Nathan and Lee, Nayeon and Krakover, Neta Gur-Ari and Cameron, Nicholas and Roberts, Nicholas and Doiron, Nick and Nangia, Nikita and Deckers, Niklas and Muennighoff, Niklas and Keskar, Nitish Shirish and Iyer, Niveditha S. and Constant, Noah and Fiedel, Noah and Wen, Nuan and Zhang, Oliver and Agha, Omar and Elbaghdadi, Omar and Levy, Omer and Evans, Owain and Casares, Pablo Antonio Moreno and Doshi, Parth and Fung, Pascale and Liang, Paul Pu and Vicol, Paul and Alipoormolabashi, Pegah and Liao, Peiyuan and Liang, Percy and Chang, Peter and Eckersley, Peter and Htut, Phu Mon and Hwang, Pinyu and Miłkowski, Piotr and Patil, Piyush and Pezeshkpour, Pouya and Oli, Priti and Mei, Qiaozhu and Lyu, Qing and Chen, Qinlang and Banjade, Rabin and Rudolph, Rachel Etta and Gabriel, Raefer and Habacker, Rahel and Delgado, Ramón Risco and Millière, Raphaël and Garg, Rhythm and Barnes, Richard and Saurous, Rif A. and Arakawa, Riku and Raymaekers, Robbe and Frank, Robert and Sikand, Rohan and Novak, Roman and Sitelew, Roman and LeBras, Ronan and Liu, Rosanne and Jacobs, Rowan and Zhang, Rui and Salakhutdinov, Ruslan and Chi, Ryan and Lee, Ryan and Stovall, Ryan and Teehan, Ryan and Yang, Rylan and Singh, Sahib and Mohammad, Saif M. and Anand, Sajant and Dillavou, Sam and Shleifer, Sam and Wiseman, Sam and Gruetter, Samuel and Bowman, Samuel R. and Schoenholz, Samuel S. and Han, Sanghyun and Kwatra, Sanjeev and Rous, Sarah A. and Ghazarian, Sarik and Ghosh, Sayan and Casey, Sean and Bischoff, Sebastian and Gehrmann, Sebastian and Schuster, Sebastian and Sadeghi, Sepideh and Hamdan, Shadi and Zhou, Sharon and Srivastava, Shashank and Shi, Sherry and Singh, Shikhar and Asaadi, Shima and Gu, Shixiang Shane and Pachchigar, Shubh and Toshniwal, Shubham and Upadhyay, Shyam and Shyamolima,  and {Debnath} and Shakeri, Siamak and Thormeyer, Simon and Melzi, Simone and Reddy, Siva and Makini, Sneha Priscilla and Lee, Soo-Hwan and Torene, Spencer and Hatwar, Sriharsha and Dehaene, Stanislas and Divic, Stefan and Ermon, Stefano and Biderman, Stella and Lin, Stephanie and Prasad, Stephen and Piantadosi, Steven T. and Shieber, Stuart M. and Misherghi, Summer and Kiritchenko, Svetlana and Mishra, Swaroop and Linzen, Tal and Schuster, Tal and Li, Tao and Yu, Tao and Ali, Tariq and Hashimoto, Tatsu and Wu, Te-Lin and Desbordes, Théo and Rothschild, Theodore and Phan, Thomas and Wang, Tianle and Nkinyili, Tiberius and Schick, Timo and Kornev, Timofei and Telleen-Lawton, Timothy and Tunduny, Titus and Gerstenberg, Tobias and Chang, Trenton and Neeraj, Trishala and Khot, Tushar and Shultz, Tyler and Shaham, Uri and Misra, Vedant and Demberg, Vera and Nyamai, Victoria and Raunak, Vikas and Ramasesh, Vinay and Prabhu, Vinay Uday and Padmakumar, Vishakh and Srikumar, Vivek and Fedus, William and Saunders, William and Zhang, William and Vossen, Wout and Ren, Xiang and Tong, Xiaoyu and Zhao, Xinran and Wu, Xinyi and Shen, Xudong and Yaghoobzadeh, Yadollah and Lakretz, Yair and Song, Yangqiu and Bahri, Yasaman and Choi, Yejin and Yang, Yichi and Hao, Yiding and Chen, Yifu and Belinkov, Yonatan and Hou, Yu and Hou, Yufang and Bai, Yuntao and Seid, Zachary and Zhao, Zhuoye and Wang, Zijian and Wang, Zijie J. and Wang, Zirui and Wu, Ziyi},
  title = {Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""

_DESCRIPTION = """\
The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to
probe large language models, and extrapolate their future capabilities.
"""

_HOMEPAGE = "https://github.com/google/BIG-bench"

_LICENSE = "Apache License 2.0"


def div_or_none(x, y):
    return x // y if x else x


def validate_task_name(task_name: str) -> None:
    """Check that the requested task name is a valid bigbench json task."""
    if task_name in bb_utils.get_all_json_task_names():
        return
    elif task_name in bb_utils.get_all_programmatic_task_names():
        raise ValueError(
            "BIG-Bench does not support programmatic tasks through HuggingFace datasets"
            f"Please see {_HOMEPAGE} for more information for how to interact with the programmatic tasks."
        )
    else:
        raise ValueError(
            f"Invalid task_name. Got task_name = {task_name}. Please choose one from:\n -- "
            + "\n -- ".join(bb_utils.get_all_json_task_names())
        )


def validate_subtask_name(task_name: str, subtask_name: str) -> None:
    """Check that the requested subtask name is a valid bigbench subtask."""
    subtasks = [name.split(":")[-1] for name in bb_utils.get_subtask_names_from_task(task_name)]
    if not subtasks:
        raise ValueError(f"Task {task_name} has no subtasks. Got subtask_name = {subtask_name}.")
    elif subtask_name not in subtasks:
        raise ValueError(
            f"Invalid subtask_name {subtask_name} for task {task_name}. Please choose one from:\n -- "
            + "\n -- ".join(subtasks)
        )


class BigBenchConfig(datasets.BuilderConfig):
    def __init__(
        self,
        name,
        subtask_name: Optional[str] = None,
        num_shots: int = 0,
        max_examples: Optional[int] = None,
        **kwargs,
    ):
        if subtask_name is not None:
            name += f"_subtask={subtask_name}"
        if num_shots != 0:
            name += f"_num_shots={num_shots}"
        if max_examples is not None:
            name += f"_max_examples={max_examples}"
        super().__init__(
            name=name,
            **kwargs,
        )
        """BIG-bench configuration.

        Args:
          name: BIG-bench task name.
          subtask_name: BIG-bench subtask name. Accepts both "task_name:subtask_name" and "subtask_name" formats.
          num_shots: Number of few-shot examples in input prompt. Default is zero.
          max_examples: Limit number of examples for each task. Default is including all examples.
        """
        self.task_name = name
        self.subtask_name = subtask_name
        self.num_shots = num_shots
        self.max_examples = max_examples


class Bigbench(datasets.GeneratorBasedBuilder):
    """The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark
    intended to probe large language models, and extrapolate their future capabilities."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = BigBenchConfig

    BUILDER_CONFIGS = [
        BigBenchConfig(name=name, version=datasets.Version("1.0.0")) for name in bb_utils.get_all_json_task_names()
    ]

    def _info(self):
        features = datasets.Features(
            {
                "idx": datasets.Value("int32"),
                "inputs": datasets.Value("string"),
                "targets": datasets.Sequence(datasets.Value("string")),
                "multiple_choice_targets": datasets.Sequence(datasets.Value("string")),
                "multiple_choice_scores": datasets.Sequence(datasets.Value("int32")),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.splits.NamedSplit("default"),  # TODO(ajandreassen): Is there a way to call this 'all'?
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "all",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(
        self,
        split,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        validate_task_name(self.config.task_name)
        if self.config.subtask_name:
            # Subtasks are sometimes in bigbench written as task_name:subtask_name.
            # We want to remove the task_name from the subtask names:
            self.config.subtask_name = self.config.subtask_name.split(":")[-1]
            validate_subtask_name(self.config.task_name, self.config.subtask_name)

        """Yields examples as (key, example) tuples."""
        if split == "all":
            # not cutoff in number of examples for 'all' split
            MIN_VALIDATION_EXAMPLES = 0
        else:
            MIN_VALIDATION_EXAMPLES = 16

        try:
            task_path, json_util = bb_json_paths.get_task_path(self.config.task_name)

            has_subtasks = bb_json_paths.has_subtasks(self.config.task_name)
            if has_subtasks:
                subtask_names = bb_json_paths.get_subtask_names(self.config.task_name)
                num_subtasks = len(subtask_names)
                min_validation_examples_per_subtask = div_or_none(MIN_VALIDATION_EXAMPLES, num_subtasks)

            if not has_subtasks:
                ds_fn = bbb.get_dataset_fn(
                    task_name=self.config.task_name,
                    task_path=task_path,
                    subtask_name=None,
                    num_shots=self.config.num_shots,
                    bigbench_task_type=bbb.BigBenchTaskType.HUGGINGFACE,
                    max_examples=self.config.max_examples,
                    json_util=json_util,
                    min_validation_examples=MIN_VALIDATION_EXAMPLES,
                    format_fn=json_task.default_format_fn,
                )
                ds_list = [ds_fn(split)]
            elif self.config.subtask_name is not None:
                ds_fn = bbb.get_dataset_fn(
                    task_name=self.config.task_name,
                    task_path=task_path,
                    subtask_name=self.config.subtask_name,
                    num_shots=self.config.num_shots,
                    bigbench_task_type=bbb.BigBenchTaskType.HUGGINGFACE,
                    max_examples=self.config.max_examples,
                    json_util=json_util,
                    min_validation_examples=min_validation_examples_per_subtask,
                    format_fn=json_task.default_format_fn,
                )
                ds_list = [ds_fn(split)]
            else:
                # Create mixture of all subtasks
                ds_list = []
                for subtask_name in subtask_names:
                    subtask_name = subtask_name.split(":")[-1]
                    logger.info(f"Loading subtask {split} split", subtask_name)
                    ds_fn = bbb.get_dataset_fn(
                        task_name=self.config.task_name,
                        task_path=task_path,
                        subtask_name=subtask_name,
                        num_shots=self.config.num_shots,
                        bigbench_task_type=bbb.BigBenchTaskType.HUGGINGFACE,
                        max_examples=div_or_none(self.config.max_examples, num_subtasks),
                        json_util=json_util,
                        min_validation_examples=min_validation_examples_per_subtask,
                        format_fn=json_task.default_format_fn,
                    )
                    ds_list.append(ds_fn(split))
        except ValueError as value_error:
            # BIG-Bench requires at least 16 examples to use the train & validation splits,
            # while using 'all'/'default' does not have such a requirement.
            if "has too few examples" in value_error.args[0] and split != "all":
                logger.warning(
                    f"-- WARNING: skipping split {split} because it has too few examples. Please use 'default' split."
                )
                logger.warning(value_error)
                return
            raise value_error

        unique_key_counter = 0
        for ds in ds_list:
            for example in ds:
                unique_key_counter += 1
                yield unique_key_counter, {
                    "idx": example["idx"],
                    "inputs": example["inputs"].numpy().decode().strip(),
                    "targets": [target.numpy().decode().strip() for target in example["targets"]],
                    "multiple_choice_targets": [
                        targets.decode().strip() for targets in example["multiple_choice_targets"].numpy()
                    ],
                    "multiple_choice_scores": [scores for scores in example["multiple_choice_scores"].numpy()],
                }
