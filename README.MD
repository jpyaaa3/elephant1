# elephant1

ArUco 마커 기반 위치 추정 및 로봇 제어 시스템



## 📢 알립니다

https://github.com/jpyaaa3/error_collector

↑ 각 기능들을 분리해서 성능 이슈를 회피한 버전입니다 ↑

▶ 엔코더 및 IMU값 수집

▶ 아루코 마커 수집

▶ 각 수집 결과 취합

▶ 취합 결과 리플레이

## 🐳 Docker 실행 방법

### 기본 실행

```bash
docker compose up -d
```

* 컨테이너만 실행됨
* 카메라 / USB 장치 접근 없음



### 카메라 및 USB 장치 사용 시 (권장)

```bash
docker compose -f docker-compose.yml -f docker-compose.serial.yml -f docker-compose.cam.yml up -d
```

✔ 사용 장치:

* `/dev/ttyUSB0`
* `/dev/video0`

✔ 특징:

* 장치가 나중에 연결되어도 접근 가능
* 장치가 없어도 에러 없이 실행됨

👉 카메라 또는 USB 장치를 사용할 가능성이 있으면 이 방식 권장



## 📂 aruco_pose

* `lite.sh`
  → 기록(recording)에 필요한 최소 프로세스만 실행

* `run_all.sh`
  → 전체 프로세스 실행

아마 얘도 작동 안 할 수도 있는데... 빠르게 고치겠습니다



## 🐘 elesim

elesim(elephant simulator) 완성됐습니다. wip은 이제 obsolete입니다!

자잘하게 수정할 건 things2fix.txt에 적어놓긴 했는데...

일단 go2 안 붙인 선에서 가장 개선된 거라고 보시면 될 것 같습니다.

! aruco_pose와는 별개로 순수 simulator입니다 !


## 💬 Notes

* Docker 기반 실행 환경 권장
* 장치 사용 여부에 따라 compose 옵션 선택
