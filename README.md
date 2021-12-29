# HCFD Night Duty Planning Model
<br>

## A Linear Programming Model for Fire Station Job Assignment Automation

消防隊的夜間固定勤務，包含值班（又稱「值宿」）、緊急救護（下稱「大夜救護」）。每日由一名隊員值宿、兩名隊員擔任大夜救護。<br>
此自動化模型根據每位隊員的休假狀況、每月應負責夜間勤務的次數，規劃每天的值宿和大夜救護人員。<br><br>
詳細圖文說明，請[點此看投影片](https://docs.google.com/presentation/d/1OPs6KNX_HXDKW3UWI20ocvyTWSONn_0s7FSHzhRzW-Y)
<br><br>


## Requirements
Python 3, PuLP, Numpy, Pandas, tqdm.
<br><br>


## Quick Start
### 1. Prepare Your Data
- A pre-defined Excel file template is provided [here on GitHub](https://github.com/DMC13/HCFD_NDP-model/blob/main/input_template.xlsx). You may also download the file [here from Google Drive](https://docs.google.com/spreadsheets/d/134gZbdU7QMkgSNCkhDDadNP0Ex56Q1nD).
- You may customize your own data format, and then modify the `prepare_input()` function in `utils.py`.
<br>

### 2. Run the Scripts
There are two ways to execute the code.<br>
1.  With command line:
```
# clone this repository
!git clone https://github.com/DMC13/HCFD_NDP-model.git

# execute
!cd HCFD_NDP-model
!python night_duty_planning.py --file_path='{your filename}.xlsx {add other kwargs here}'
```
2.  On Colab, see the example [here](https://colab.research.google.com/drive/18Yhb-QnT3Pc_bTsfHjqUXU3dJw_Z6nv_?usp=sharing).
<br><br>


## Citation
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
Special thanks to:<br>
- 廖宜冠, Branch Head, Minghu Station, Hsinchu City Fire Department.
- 王宥程, Squad Leader, Minghu Station, Hsinchu City Fire Department.
- PJ Huang, Substitute Civilian Serviceman, Minghu Station, Hsinchu City Fire Department.
<br><br>


## Get Involved
You are welcome to contribute to this Open Solution. To get started:
- Check issues to see if there is something you would like to contribute to.
- Feel free to fork this repository or provide pull request.
