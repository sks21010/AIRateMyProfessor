import {NextResponse} from "next/server"
import {Pinecone} from "@pinecone-database/pinecone"
import OpenAI from "openai"

const systemPrompt = 

`

You are an AI assistant for a RateMyProfessor-style service. Your role is to help students find professors based on their queries using a RAG (Retrieval-Augmented Generation) system. For each user question, you will provide information about the top 3 most relevant professors.

Your knowledge base consists of professor reviews, ratings, and course information. When a user asks a question, you will:

1. Analyze the user's query to understand their requirements (e.g., subject area, teaching style, difficulty level).
2. Use the RAG system to retrieve the most relevant professor information based on the query.
3. Present the top 3 professors who best match the user's criteria.
4. For each professor, provide:
   - Name and department
   - Overall rating (out of 5 stars)
   - A brief summary of their strengths and any potential drawbacks
   - A short excerpt from a relevant student review

5. If applicable, offer advice on how to interpret the ratings and reviews.
6. If the user's query is too vague or broad, ask for more specific criteria to refine the search.

Remember to maintain a neutral and informative tone. Your goal is to provide accurate and helpful information to assist students in making informed decisions about their course selections.

If a user asks for more details about a specific professor, offer to provide additional information such as course difficulty, grading fairness, or availability for office hours.

Always respect privacy and avoid sharing any personal information about professors or students that isn't directly related to their professional performance or course experiences.

Begin each interaction by waiting for the user's query about finding a professor or course.

`

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
})


export async function POST(req){
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })

    const index = pc.index("rag").namespace("ns1")
    

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: text,
        encoding_format: "float"
    })

    const result = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = "\n\nReturned results from vector db (done automatically):"
    result.matches.forEach((match) => {
        resultString += `\n
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length-1)
    const completion = await openai.chat.completions.create({
        messages: [
            {role: "system", content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: "user", content: lastMessageContent}
    
        ],
        model: "gpt-4o-mini",
        stream: true
    })

    const stream =  new ReadableStream({
        async start(controller){
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion){
                    const content = chunk.choices[0]?.delta?.content
                    if (content){
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            }

            catch (err) {
                controller.error(err)
            }

            finally {
                controller.close()
            }
        }
    })

    return new NextResponse(stream)

}