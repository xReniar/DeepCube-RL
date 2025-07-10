class DummyCube():
    def __init__(self):
        self.top = list("....U....")
        self.bottom = list(".D.DDD.D.")
        self.front = list("....F..F.")
        self.right = list("....R..R.")
        self.back = list("....B..B.")
        self.left = list("....L..L.")

    def face_rotation(self, face_list:str):
        copy = face_list[:]
        # Applica la rotazione
        face_list[0] = copy[6]
        face_list[1] = copy[3]
        face_list[2] = copy[0]
        face_list[3] = copy[7]
        face_list[4] = copy[4]
        face_list[5] = copy[1]
        face_list[6] = copy[8]
        face_list[7] = copy[5]
        face_list[8] = copy[2]


    def U(self, clockwise: bool, n_times: int):
        n = 1 if clockwise else 3
        n = 2 if n_times == 2 else n
        for _ in range(n):
            self.face_rotation(self.top)

            f_copy = self.front[:3]
            r_copy = self.right[:3]
            b_copy = self.back[:3]
            l_copy = self.left[:3]

            for i in range(3):
                self.front[i] = r_copy[i]
                self.right[i] = b_copy[i]
                self.back[i] = l_copy[i]
                self.left[i] = f_copy[i]

    def D(self, clockwise: bool, n_times: int):
        n = 1 if clockwise else 3
        n = 2 if n_times == 2 else n
        for _ in range(n):
            self.face_rotation(self.bottom)

            f_copy = self.front[6:]
            r_copy = self.right[6:]
            b_copy = self.back[6:]
            l_copy = self.left[6:]

            for i in range(6, 9):
                self.front[i] = l_copy[i-6]
                self.right[i] = f_copy[i-6]
                self.back[i] = r_copy[i-6]
                self.left[i] = b_copy[i-6]

    def F(self, clockwise: bool, n_times: int):
        n = 1 if clockwise else 3
        n = 2 if n_times == 2 else n
        for _ in range(n):
            self.face_rotation(self.front)

            top, right, bottom, left = [None] * 3, [None] * 3, [None] * 3, [None] * 3
            for i in range(3):
                top[i] = self.top[i+6]
                right[i] = self.right[i*3]
                bottom[i] = self.bottom[2-i]
                left[i] = self.left[8-i*3]

            for i in range(3):
                self.top[6 + i] = left[i]
                self.right[i * 3] = top[i]
                self.bottom[2 - i] = right[i]
                self.left[8-i*3] = bottom[i]

    def R(self, clockwise: bool, n_times: int):
        n = 1 if clockwise else 3
        n = 2 if n_times == 2 else n
        for _ in range(n):
            self.face_rotation(self.right)

            top, back, bottom, front = [None] * 3, [None] * 3, [None] * 3, [None] * 3
            for i in range(2, 9, 3):
                x = int((i - 2)/3)
                top[x] = self.top[i]
                front[x] = self.front[i]
                bottom[x] = self.bottom[i]
                back[x] = self.back[8-i]

            for i in range(2, 9, 3):
                x = int((i - 2)/3)
                self.top[i] = front[x]
                self.front[i] = bottom[x]
                self.bottom[i] = back[x]
                self.back[8-i] = top[x]

    def B(self, clockwise: bool, n_times: int):
        n = 1 if clockwise else 3
        n = 2 if n_times == 2 else n
        for _ in range(n):
            self.face_rotation(self.back)

            top, right, bottom, left = [None] * 3, [None] * 3, [None] * 3, [None] * 3
            for i in range(3):
                top[i] = self.top[i]
                right[i] = self.right[2+(i*3)]
                bottom[i] = self.bottom[8-i]
                left[i] = self.left[6-i*3]

            for i in range(3):
                self.top[i] = right[i]
                self.right[2+(i * 3)] = bottom[i]
                self.bottom[8 - i] = left[i]
                self.left[6-i*3] = top[i]

    def L(self, clockwise: bool, n_times: int):
        n = 1 if clockwise else 3
        n = 2 if n_times == 2 else n
        for _ in range(n):
            self.face_rotation(self.left)

            top, back, bottom, front = [None] * 3, [None] * 3, [None] * 3, [None] * 3
            for i in range(0, 7, 3):
                x = int(i/3)
                top[x] = self.top[i]
                front[x] = self.front[i]
                bottom[x] = self.bottom[i]
                back[x] = self.back[8-i]

            for i in range(0, 7, 3):
                x = int(i/3)
                self.top[i] = back[x]
                self.front[i] = top[x]
                self.bottom[i] = front[x]
                self.back[8-i] = bottom[x]

    def rotate(self, move: str):
        clockwise = (len(move) == 1)
        n_turns = 2 if not clockwise and move[1].isdigit() else 1

        if move[0] == "U":
            self.U(clockwise, n_turns)
        if move[0] == "D":
            self.D(clockwise, n_turns)
        if move[0] == "F":
            self.F(clockwise, n_turns)
        if move[0] == "R":
            self.R(clockwise, n_turns)
        if move[0] == "B":
            self.B(clockwise, n_turns)
        if move[0] == "L":
            self.L(clockwise, n_turns)
    
    def __str__(self):
        def format_face(face):
            return [face[i*3:(i+1)*3] for i in range(3)]

        top = format_face(self.top)
        bottom = format_face(self.bottom)
        left = format_face(self.left)
        front = format_face(self.front)
        right = format_face(self.right)
        back = format_face(self.back)

        lines = []

        for row in top:
            lines.append("      " + " ".join(row))

        for i in range(3):
            lines.append(" ".join(left[i]) + " " +
                        " ".join(front[i]) + " " +
                        " ".join(right[i]) + " " +
                        " ".join(back[i]))

        for row in bottom:
            lines.append("      " + " ".join(row))

        return "\n".join(lines)

    def get_kociemba_facelet_positions(self) -> str:
        u = "".join(self.top)
        r = "".join(self.right)
        f = "".join(self.front)
        d = "".join(self.bottom)
        l = "".join(self.left)
        b = "".join(self.back)

        return f"{u}{r}{f}{d}{l}{b}"