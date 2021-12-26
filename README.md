# HCFD Night Duty Planning Model

消防隊的夜間固定勤務，包含值班（又稱「值宿」）、緊急救護（下稱「大夜救護」）。每日由一名隊員值宿、兩名隊員擔任大夜救護。<br>
此自動化模型根據每位隊員的休假狀況、每月應負責夜間勤務的次數，規劃每天的值宿和大夜救護人員。
<br><br>


## Requirements
---
Python 3.x, PuLP, Numpy, Pandas, tqdm.
<br><br>


## Quick Start
---
### 1. Prepare Your Data
- A pre-defined excel file format is provided [here](https://github.com/DMC13/HCFD_NDP-model/input_template.xlsx).
- You may define your own input format, and then modify the `prepare_input()` function in `utils.py`.
<br>

### 2. Run the Scripts
There are two ways to execute the code.<br>
1.  Run from command line:
```
# clone this repository
!git clone --quiet https://github.com/DMC13/HCFD_NDP-model.git
# execute
!python night_duty_planning.py --file_path='{your filename}.xlsx {add other kwargs here}'
```
2.  Run on Colab, see the example [here 待新增](https://colab.ressearch.google.com)
<br><br>

## Citation
---
Use this bibtex to cite this repository:
```
@misc{DMC13_HCFD_NDP-model_2021,
  title={Night Duty Planning: A Linear Programming Model for Fire Station Job Assignment Automation},
  author={Tao-Ming Chen},
  year={2021},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/DMC13/HCFD_NDP-model}},
}
```
<br>


## Acknowledgement
---
Special thanks to:<br>
- 廖宜冠, Branch Head, Minghu Station, Hsinchu City Fire Department.
- 王宥程, Squad Leader, Minghu Station, Hsinchu City Fire Department.
- PJ Huang, Substitute Civilian Serviceman, Minghu Station, Hsinchu City Fire Department.