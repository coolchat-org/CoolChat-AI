import http from "k6/http";
import { group, check, sleep } from "k6";

export let options = {
  vus: 90,
  duration: "1m",
};
// export const options = {
//     stages: [
//         { duration: '10s', target: 40 },
//         { duration: '20s', target: 90 },
//         { duration: '10s', target: 0 },
//     ],
//     // thresholds: {
//     //     http_req_duration: ['p(95)<3000'], // 95% requests phải dưới 1s
//     //     http_req_failed: ['rate<0.05'],    // Lỗi không quá 5%
//     // }
// };

const BASE_URL = "https://coolchatai.onrender.com";
const ORG_ID = "abcv";
const defaultHeaders = {
  "Content-Type": "application/json",
  "Accept": "application/json",
};

export default function () {
  group("Create new chat and send message", () => {
    // B1: Gọi new-chat để tạo chat mới
    let resNewChat = http.put(`${BASE_URL}/v2/chat/new-chat`, null, {
      headers: defaultHeaders,
    });

    check(resNewChat, {
      "new chat created": (r) => r.status === 200,
    });

    let chatId;
    try {
      chatId = resNewChat.json()._id; // hoặc resNewChat.json().id nếu key là vậy
    } catch (err) {
      console.error("Không lấy được chat ID từ response");
      return;
    }

    sleep(0.5);

    // B2: Gửi message tới chat mới tạo
    const messageBody = {
      new_message: "Hello from K6",
      index_host: "coolindex-oei5fcs.svc.aped-4627-b74a.pinecone.io",
      namespace: "coolindex-84e73dc50f2be900",
      virtual_namespace: null,
      config: {
        chatbot_attitude: "friendly",
        company_name: "CoolChat",
        start_sentence: "Chào bạn chúng tôi là công ty tư vấn số 1 vịnh bắc bộ!",
        end_sentence: "Hẹn gặp lại quý khách vào lần sau!",
      },
    };

    let resReply = http.post(`${BASE_URL}/v2/chat/${chatId}`, JSON.stringify(messageBody), {
      headers: defaultHeaders,
    });

    check(resReply, {
      "reply message sent": (r) => r.status === 200,
    });

    sleep(1); // tránh DDoS
  });
}
