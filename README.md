# OUTLINE

train_data/storage 폴더에는 이번 프로젝트에서 학습한 모델이 거의 다 담겨있습니다.

폴더에 옮기지 않고 바로 수정한 부분도 있어서 모두 반영되어 있지는 않으니 주의해 주십시오.

용량문제로 인해 후반 train data의 training csv파일은 없습니다. 

실폐한 코드 혹은 문제가 있어 버린 코드는 failed 폴더에 넣어놨습니다.

rocket_modeling_reference는 초기 로켓 모델 구성시 참고한 코드입니다. 그러나 두 프로젝트가 자향하는 바가 다르므로 많은 부분에서 차이가 있습니다.
#https://github.com/cmontalvo251/aerospace/blob/main/rockets/rocket_seminar_series/two_stage_rocket_w_aerodynamics.py

- rocket.py는 로켓 env 구성하는 파일입니다
- test_rocket.py는 로켓 env가 잘 작동하는지 확인하는 파일입니다. (원하는 action값을 넣어 결과를 얻을 수 있습니다.)
- train.py는 학습시 실행하는 파일입니다. 학습 모델의 구성이 들어있습니다. episode를 바꿔가며 실행하면 됩니다. 파일내 Path라는 변수에 저장 경로가 지정되어 있습니다.
- trina.ipynb는 colab에서 구동시 사용하는 파일이지만 colab의 runtime문제로 업데이트 안 한지 꽤 되어 실행에 문재가 있을 수 있습니다.
- test_model은 학습한 모델을 가져와 실행시켜보는 파일입니다.  파일내 Path라는 변수에 load 경로가 지정되어 있습니다.
- train_data_visualizaion.py는 training_data.csv 파일을 시각화해주는 파일입니다.

# action space
- 5개의 엔진 x축방향 각속도 (-30~30)
- 5개의 엔진 y축방향 각속도 (-30~30)
- 5개의 엔진 추력(min~max) => continuous action space
- 3개의 엔진 x축방향 각속도 (-30~30)
- 3개의 엔진 y축방향 각속도 (-30~30)
- 3개의 엔진 추력(min~max) => continuous action space
~~- 1개의 분리(버튼)  값이 1이상이면 1단분리, 값이 2이상이면 2단 분리~~ 5.26 1553 삭제 
=> 시간이나 현재 연료 잔량에 따라 분리되도록 설정
=> 총 24개의 값을 list로 return


# state space
- (x, y, z 좌표)    np.array:3개 element
- (x, y, z 속도)    np.array:3개 element
- (회전 각도)        np.array:3개 element
- (회전 각속도)     np.array:3개 element
- 현재 연료 질량
- 현재 stage
- 선체를 기준으로 노즐 각도 ((0theta0, 0theta1), (1theta0, 1theta1), (2theta0, 2theta1), (3theta0, 3theta1), ...) 2D array: 2*8개 element
    theta0 = 0\~max
    theta1 = 0\~2pi
- 현재 분리 중인지 나타내는 상태 (2초가 될때까지 엔진이 작동하지 않게)
- 현재 고장 상태 : 1개의 int (0번은 고장나지 않은 상태, 1번부터 순서대로 어떤 엔진이 고장났는지 명시) 5.26 1553추가

=> 32개의 element를 state에 저장
=> 9개의 요소를 list로 저장

---

5.9 김병규 작업한 부분:
rocket.py : 
- method : init, step, get_New_state, get_aerofriction, get_gravity, get_propulsion, step 
- 부족한 부분 : 
1. 실제로 동작하는 여부는 테스트 하지 못함. 
2. get_New_state에서 coordinate 변화를 고려하지 않음. 따라서 6DOF라는 모듈사용해서 coordinate 변화 계산하기 편하게 바꿀 거임.


5.11 남윤호 작업
rocket.py
- 상수값 채우기
- 부족한 부분: TODO 참조


5.13 남민영 작업
rocket.py
- flatten 함수 제작, create_initial_state 수정
- 부족한 부분: 수정된 create_initial_state에 맞춰 get_new_state 수정 필요(y축 도입)


5.14 김병규 작업
rocket.py
- dynamics 검토 후 최고속도 O km/s 으로 나타나는 것 확인 => stage 분리작업도 추후 확인 예정
- animation 변경
test_rocket.py
- rocket 변경 후 세부사항 수정


5.18 남윤호 작업
rocket.py
- rocket.mass 수정, 관성모멘트 공식 도입
- TODO: mass에 따라 관성모멘트가 달라지는 부분도 step에 넣어야 할 듯
- 월요일 수업 전까지 policy.py 검토 예정


5.19 남민영 작업
rocket.py
- calculate_reward 함수 마무리, check_crash 함수 수정
- TODO: target_p (목표 궤도) 설정
policy.py
- ActorCritic 참고 자료: https://wikidocs.net/172977
test_train.py 생성


5.20 남윤호 작업
policy.py
- 함수 검토 완료
- 변수들의 파라미터화를 통해 테스트를 조금 더 쉽도록 함


5.20 남민영 작업
policy.py
- 함수 검토 및 주석 추가


5.22 남민영 작업
rocket.py
- target_p 변수 추가 및 calculate_reward 수정
- target_p 참고 자료: https://forum.nasaspaceflight.com/index.php?action=dlattach;topic=29214.0;attach=489078


5.23-24 김병규 작업
- train.ipynb 작업 (colab에서 학습환경 구성, ppo base로 기본적인 학습 확인)


5.26 김병규 작업
- action에서 분리 작업 삭제, 자동분리 구현, 분리시간(2초) 구현=> state에 값 1개 더 추가함.(분리 중에는 action과 상관없이 추력이 0으로 고정) 
- trajectory method 변경

=>>>여기다가 본인이 작업한 부분 추가해서 넣으면 될 듯합니다.

5.27 남윤호 작업
- train.ipynb 작업(reward plot, action csv파일로 저장)
- 여전히 학습이 잘 안되는 부분을 확인함

5.27 남민영 작업
rocket.py
- 1.1 검토 및 1.2(소모량 대비 고도 보상) 추가

5.26-6.2 김병규 작업
- 계속 학습하고 수정 반봅

---

# What we need to: 
(구현 필요해 보이는 부분 추가하기, 완료했으면 취소선으로 변경)


 ~~* flatten 함수 구현 : line45~~ 


 ~~7개의 요소의 list를 input으로 받아서 30개의 element를 갖는 list로 풀어주는 함수~~ 


~~이 함수를 구현해야 나중에 NN에 state를 input으로 집어넣기 쉬울 것임.~~

* ~~기본적인 변수들을 실제값으로 변경.~~
* ~~그래프 그리는 method 만들고 실제 동작 확인.~~
* ~~calculate_reward 부분 구현.~~
* ~~policy.py 검토 및 전체적인 RL 구성하기~~
* ~~visualization: 제대로 학습 되었는지 확인하는 부분 첨가~~
* ~~get_new_state: y좌표 계산 필요~~

