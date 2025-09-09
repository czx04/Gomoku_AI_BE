import math, random
import os
import numpy as np
import h5py
import re
# --- Config ---
N = 20
WIN_LEN = 5
MAX_CANDIDATES = 24
NEAR_DIST = 2
SEARCH_DEPTH = 2
DIRS = [(1,0),(0,1),(1,1),(1,-1)]

# ---------------- GAME ----------------
class GomokuGame:
    def __init__(self, size=N, win_len=WIN_LEN):
        self.N = size
        self.WIN_LEN = win_len

    def inb(self, r, c):
        return 0 <= r < self.N and 0 <= c < self.N

    def check_winner(self, board):
        for r in range(self.N):
            for c in range(self.N):
                p = board[r,c]
                if p == 0: continue
                for dr,dc in DIRS:
                    cnt = 1
                    rr,cc=r+dr,c+dc
                    while self.inb(rr,cc) and board[rr,cc]==p:
                        cnt+=1; rr+=dr; cc+=dc
                        if cnt>=self.WIN_LEN: return p
        return 0

    def is_full(self, board): return not (board==0).any()
    def has_any_stone(self, board): return (board!=0).any()

    def legal_moves_mask(self, board):
        return (board==0).astype(np.float32).reshape(-1)

    def encode_board(self, board):
        x = np.zeros((self.N,self.N,3), dtype=np.float32)
        x[:,:,0] = (board==1).astype(np.float32)
        x[:,:,1] = (board==-1).astype(np.float32)
        x[:,:,2] = (board==0).astype(np.float32)
        return x

    def random_board(self, max_plies=40):
        b = np.zeros((self.N,self.N), dtype=np.int8)
        turn = 1
        plies = random.randint(0, max_plies)
        for _ in range(plies):
            if self.check_winner(b)!=0 or self.is_full(b): break
            empties = np.argwhere(b==0)
            if len(empties)==0: break
            r,c = map(int, random.choice(empties))
            b[r,c]=turn
            turn = -turn
        # đảm bảo lượt X
        x_cnt = int((b==1).sum()); o_cnt = int((b==-1).sum())
        if x_cnt>o_cnt:
            empties = np.argwhere(b==0)
            if len(empties)>0 and self.check_winner(b)==0:
                r,c = map(int, random.choice(empties))
                b[r,c] = -1
        return b

# ---------------- HEURISTIC ----------------
class Heuristic:
    def __init__(self, game: GomokuGame):
        self.game = game

    def seq_score(self, k, open_ends):
        if k>=self.game.WIN_LEN: return 1e7
        base={1:2,2:12,3:80,4:600}.get(k,0)
        bonus={0:0,1:1.0,2:2.5}.get(open_ends,0.0)
        return base*bonus

    def eval_line(self, line, player):
        opp=-player
        L=len(line); s=0.0; i=0
        while i<L:
            if line[i]==player:
                j=i
                while j<L and line[j]==player: j+=1
                k=j-i
                left_block = (i-1>=0 and line[i-1]==opp)
                right_block = (j<L and line[j]==opp)
                open_ends = 2 - int(left_block) - int(right_block)
                s += self.seq_score(k, max(0, open_ends))
                i=j
            else:
                i+=1
        return s

    def eval_player(self, board, player):
        N=self.game.N
        s=0.0
        for r in range(N): s += self.eval_line(board[r,:], player)
        for c in range(N): s += self.eval_line(board[:,c], player)
        for k in range(-(N-1), N):
            s += self.eval_line(np.diag(board, k=k), player)
        fl = np.fliplr(board)
        for k in range(-(N-1), N):
            s += self.eval_line(np.diag(fl, k=k), player)
        return s

    def local_heuristic(self, board, r, c, player):
        s=0.0
        for dr,dc in DIRS:
            line=[]
            for t in range(-self.game.WIN_LEN+1, self.game.WIN_LEN):
                rr,cc=r+dr*t, c+dc*t
                if self.game.inb(rr,cc): line.append(board[rr,cc])
            if line:
                arr = np.array(line, dtype=np.int8)
                s += self.eval_line(arr, player)
        return s

    def evaluate(self, board):
        win = self.game.check_winner(board)
        if win==1: return 1e9
        if win==-1: return -1e9
        return self.eval_player(board,1) - self.eval_player(board,-1)

# ---------------- ALPHA-BETA ----------------
class AlphaBeta:
    def __init__(self, game: GomokuGame, heuristic: Heuristic, depth=SEARCH_DEPTH):
        self.game = game
        self.h = heuristic
        self.depth=depth

    def candidate_moves(self, board):
        if not self.game.has_any_stone(board):
            return [(self.game.N//2, self.game.N//2)]
        empties = np.argwhere(board==0)
        occ = np.argwhere(board!=0)
        occ_set = set(map(tuple, occ))
        goods=[]
        for r,c in empties:
            for rr,cc in occ_set:
                if abs(rr-r)<=NEAR_DIST and abs(cc-c)<=NEAR_DIST:
                    goods.append((r,c)); break
        if len(goods)<=MAX_CANDIDATES: return goods
        scored=[]
        for (r,c) in goods:
            board[r,c]=1
            s1=self.h.local_heuristic(board,r,c,1)
            board[r,c]=-1
            s2=self.h.local_heuristic(board,r,c,-1)
            board[r,c]=0
            scored.append(((r,c),max(s1,s2)))
        scored.sort(key=lambda x:x[1],reverse=True)
        return [pos for pos,_ in scored[:MAX_CANDIDATES]]

    def search(self, board, depth, alpha, beta, maximizing):
        win=self.game.check_winner(board)
        if depth==0 or win!=0 or self.game.is_full(board):
            return self.h.evaluate(board), None
        moves=self.candidate_moves(board)
        if not moves: return self.h.evaluate(board), None

        scored=[]
        for (r,c) in moves:
            board[r,c]=1 if maximizing else -1
            s=self.h.local_heuristic(board,r,c,1 if maximizing else -1)
            board[r,c]=0
            scored.append(((r,c),s))
        scored.sort(key=lambda x:x[1], reverse=maximizing)
        ordered=[pos for pos,_ in scored]

        if maximizing:
            best_val=-math.inf; best_mv=None
            for (r,c) in ordered:
                board[r,c]=1
                val,_=self.search(board,depth-1,alpha,beta,False)
                board[r,c]=0
                if val>best_val: best_val,best_mv=val,(r,c)
                alpha=max(alpha,best_val)
                if beta<=alpha: break
            return best_val,best_mv
        else:
            best_val=math.inf; best_mv=None
            for (r,c) in ordered:
                board[r,c]=-1
                val,_=self.search(board,depth-1,alpha,beta,True)
                board[r,c]=0
                if val<best_val: best_val,best_mv=val,(r,c)
                beta=min(beta,best_val)
                if beta<=alpha: break
            return best_val,best_mv

    def best_move(self, board):
        _, mv=self.search(board,self.depth,-math.inf,math.inf,True)
        return mv

# ---------------- DATASET ----------------
class DatasetGenerator:
    def __init__(self, game:GomokuGame, ab:AlphaBeta):
        self.game = game
        self.ab = ab

    def _ensure_dsets(self, f):
        if "X" in f:
            return f["X"], f["y"], f["mask"]
        X_dset = f.create_dataset(
            "X", shape=(0, self.game.N, self.game.N, 3),
            maxshape=(None, self.game.N, self.game.N, 3),
            dtype="float32", chunks=True
        )
        y_dset = f.create_dataset(
            "y", shape=(0,), maxshape=(None,),
            dtype="int32", chunks=True
        )
        m_dset = f.create_dataset(
            "mask", shape=(0, self.game.N*self.game.N),
            maxshape=(None, self.game.N*self.game.N),
            dtype="float32", chunks=True
        )
        return X_dset, y_dset, m_dset

    def generate_to_h5(self, filename="gomoku_dataset.h5", target_total=20000, chunk_size=1000):
        with h5py.File(filename, "a") as f:
            X_dset, y_dset, m_dset = self._ensure_dsets(f)

            X_buf, y_buf, m_buf = [], [], []
            trials = 0

            while X_dset.shape[0] < target_total and trials < (target_total * 10):
                trials += 1
                b = self.game.random_board()
                if self.game.check_winner(b) != 0 or self.game.is_full(b):
                    continue
                mv = self.ab.best_move(b.copy())
                if mv is None:
                    continue

                idx = mv[0]*self.game.N + mv[1]
                mask = self.game.legal_moves_mask(b)
                # sanity: chỉ lấy nhãn hợp lệ
                if mask[idx] != 1.0:
                    continue

                X_buf.append(self.game.encode_board(b))
                y_buf.append(idx)
                m_buf.append(mask)

                if len(X_buf) >= chunk_size:
                    old_size = X_dset.shape[0]
                    new_size = min(old_size + len(X_buf), target_total)

                    take = new_size - old_size
                    X_dset.resize(new_size, axis=0)
                    y_dset.resize(new_size, axis=0)
                    m_dset.resize(new_size, axis=0)

                    X_dset[old_size:new_size] = np.array(X_buf[:take], dtype=np.float32)
                    y_dset[old_size:new_size] = np.array(y_buf[:take], dtype=np.int32)
                    m_dset[old_size:new_size] = np.array(m_buf[:take], dtype=np.float32)

                    X_buf, y_buf, m_buf = X_buf[take:], y_buf[take:], m_buf[take:]

            # flush nốt
            if len(X_buf) > 0 and X_dset.shape[0] < target_total:
                old_size = X_dset.shape[0]
                new_size = min(old_size + len(X_buf), target_total)
                take = new_size - old_size

                X_dset.resize(new_size, axis=0)
                y_dset.resize(new_size, axis=0)
                m_dset.resize(new_size, axis=0)

                X_dset[old_size:new_size] = np.array(X_buf[:take], dtype=np.float32)
                y_dset[old_size:new_size] = np.array(y_buf[:take], dtype=np.int32)
                m_dset[old_size:new_size] = np.array(m_buf[:take], dtype=np.float32)

            print(f"Dataset size: {X_dset.shape[0]} (after {trials} trials)")

def load_h5(filename):
    with h5py.File(filename, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
        m = f["mask"][:]
    return X,y,m

def load_or_generate_dataset(game, ab, filename="gomoku_dataset.h5", target_total=20000, chunk_size=1000):
    gen = DatasetGenerator(game, ab)
    if not os.path.exists(filename):
        gen.generate_to_h5(filename, target_total=target_total, chunk_size=chunk_size)
    else:
        # nếu file có ít hơn target_total thì bổ sung
        with h5py.File(filename, "r") as f:
            cur = f["X"].shape[0]
        if cur < target_total:
            gen.generate_to_h5(filename, target_total=target_total, chunk_size=chunk_size)
    return load_h5(filename)

# ---------------- QUICK VALIDATION ----------------
def validate_dataset(X, y, m, N=N):
    ok = True
    # shape check
    if X.ndim!=4 or X.shape[1:]!=(N,N,3): print("X shape sai"); ok=False
    if y.ndim!=1: print("y shape sai"); ok=False
    if m.ndim!=2 or m.shape[1]!=(N*N): print("mask shape sai"); ok=False
    # mask/y consistency
    bad = 0
    for i in range(len(y)):
        if m[i, y[i]] != 1.0:
            bad += 1
    if bad>0:
        print(f"{bad} labels không khớp mask (ô không trống).")
        ok=False
    # channel consistency
    ones = np.sum(X[:,:,:,0]) + np.sum(X[:,:,:,1]) + np.sum(X[:,:,:,2])
    if abs(ones - (X.shape[0]*N*N)) > 1e-3:
        print("One-hot 3 kênh không chuẩn.")
        ok=False
    if ok:
        print("Dataset OK.")
    return ok

class NpzDatasetGenerator:
    def __init__(self, game:GomokuGame, ab:AlphaBeta, output_path="dataset/generated"):
        self.game = game
        self.ab = ab
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def merge_npz_files(self,input_dir="dataset/generated", output_file="dataset/merged/gomoku_dataset.npz", delete_originals=True):
      files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npz')])

      if not files:
          print("Không tìm thấy file NPZ nào trong thư mục!")
          return 0

      all_inputs = []
      all_outputs = []
      total_positions = 0

      print(f"Bắt đầu gộp {len(files)} file NPZ...")

      # Đọc và gộp dữ liệu
      for file_name in files:
          file_path = os.path.join(input_dir, file_name)
          data = np.load(file_path)

          inputs = data['inputs']
          outputs = data['outputs']

          all_inputs.append(inputs)
          all_outputs.append(outputs)

          positions_count = inputs.shape[0]
          total_positions += positions_count

      # Gộp tất cả thành một mảng lớn
      merged_inputs = np.concatenate(all_inputs, axis=0)
      merged_outputs = np.concatenate(all_outputs, axis=0)

      # Lưu file gộp
      np.savez_compressed(output_file, inputs=merged_inputs, outputs=merged_outputs)

      # Xóa các file gốc nếu được yêu cầu
      if delete_originals:
          for file_name in files:
              file_path = os.path.join(input_dir, file_name)
              os.remove(file_path)
      return merged_inputs.shape[0]
    def add_file_index(self):
        folder_path = "dataset/generated"
        npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

        # Tìm số lớn nhất trong tên file
        max_number = -1
        for file in npz_files:
            match = re.match(r'(\d+)\.npz', file)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)
        next_number = max_number + 1
        new_filename = f"{str(next_number).zfill(5)}.npz"
        if "00000.npz" in npz_files:
            old_path = os.path.join(folder_path, "00000.npz")
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
        else:
            print("chưa có file gốc => Tạo file gốc")

    def generate_npz_files(self, target_total=20000, chunk_size=1000):
        """Tạo dataset trong định dạng NPZ tương tự code gốc"""
        file_index = 0
        total_generated = 0
        trials = 0

        while total_generated < target_total and trials < (target_total * 10):
            inputs, outputs = [], []
            chunk_generated = 0

            while chunk_generated < chunk_size and total_generated < target_total and trials < (target_total * 10):
                trials += 1

                # Tạo bàn cờ ngẫu nhiên
                b = self.game.random_board()
                if self.game.check_winner(b) != 0 or self.game.is_full(b):
                    continue

                # Tìm nước đi tốt nhất
                mv = self.ab.best_move(b.copy())
                if mv is None:
                    continue

                r, c = mv
                mask = self.game.legal_moves_mask(b)
                idx = r*self.game.N + c

                # Kiểm tra hợp lệ
                if mask[idx] != 1.0:
                    continue

                # Chuyển đổi sang định dạng một kênh như code gốc
                input_board = b.copy().reshape(self.game.N, self.game.N, 1)

                # Tạo output
                output = np.zeros((self.game.N, self.game.N), dtype=np.int8)
                output[r, c] = 1

                # Tăng cường dữ liệu
                for k in range(4):
                    input_rot = np.rot90(input_board, k=k)
                    output_rot = np.rot90(output, k=k)

                    inputs.append(input_rot)
                    outputs.append(output_rot)

                    inputs.append(np.fliplr(input_rot))
                    outputs.append(np.fliplr(output_rot))

                    inputs.append(np.flipud(input_rot))
                    outputs.append(np.flipud(output_rot))

                chunk_generated += 1
                total_generated += 1

            # Lưu chunk này
            if inputs:
                file_name = f"{str(file_index).zfill(5)}.npz"
                np.savez_compressed(
                    os.path.join(self.output_path, file_name),
                    inputs=np.array(inputs, dtype=np.int8),
                    outputs=np.array(outputs, dtype=np.int8)
                )
                file_index += 1

        print(f"Đã tạo {total_generated} vị trí trong {file_index} file (sau {trials} lần thử)")

# ---------------- USAGE ----------------




# Khởi tạo trò chơi và mô hình
game = GomokuGame()
h = Heuristic(game)
ab = AlphaBeta(game, h, depth=SEARCH_DEPTH)



# Tạo generator và sinh dữ liệu
generator = NpzDatasetGenerator(game, ab, output_path="dataset/generated")

size_file = 100
for i in range(size_file):
  generator.add_file_index()
  generator.generate_npz_files(target_total=200, chunk_size=200)
  print("Đã sinh thêm 1 file")

generator.merge_npz_files("dataset/generated", "dataset/merge")

# Hiển thị danh sách file
print("\nDanh sách file data")
for f in sorted(os.listdir(folder_path)):
    if f.endswith('.npz'):
        print(f)
