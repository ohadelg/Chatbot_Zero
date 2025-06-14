import { ChangeEvent, FormEvent, useEffect, useState } from 'react'
import { ReactComponent as RefreshIcon } from 'images/refresh_icon.svg'
import { ReactComponent as SearchIcon } from 'images/search_icon.svg'
import { ReactComponent as ArrowIcon } from 'images/arrow_icon.svg'
import { AppStatus } from 'store/provider'

export default function SearchInput({ onSearch, value, appStatus }) {
  const [query, setQuery] = useState<string>(value)
  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    if (!!query.length) {
      onSearch(query)
    }
  }
  const handleChange = (event: ChangeEvent<HTMLInputElement>) =>
    setQuery(event.target.value)

  useEffect(() => {
    setQuery(value)
  }, [value])

  return (
    <form className="w-full" onSubmit={handleSubmit} >
      <div className="relative mt-1 flex w-full items-center h-14 gap-2 flex-row-reverse">
        <input
          type="search"
          className={`hover:border-blue-500 outline-none focus-visible:border-blue-600 w-full h-14 rounded-md border-2 border-zinc-300 px-3 py-2.5 pr-12 text-base font-medium placeholder-gray-400 ${
            appStatus === AppStatus.Idle ? 'pl-20' : 'pl-40'
          }`}
          value={query}
          onChange={handleChange}
          placeholder="מה בא לך לשאול?"
        />
        <span className="pointer-events-none absolute right-4">
          <SearchIcon />
        </span>
        {appStatus === AppStatus.Idle ? (
            <button
            className="hover:bg-blue disabled:bg-blue-400 px-4 py-2 bg-blue-500 rounded flex items-center absolute left-2 z-10"
            type="submit"
            disabled={!query.length}
            >
            <span style={{ transform: 'rotate(180deg)' }}>
              <ArrowIcon width={24} height={24} />
            </span>
            </button>
        ) : (
          <button
            className="hover:bg-blue-400 hover:text-blue-100 px-4 py-2 bg-blue-100 rounded flex justify-center items-center text-blue-400 font-bold absolute left-2 z-10"
            type="submit"
          >
             התחל מחדש
            <span className="mr-2">
              <RefreshIcon width={22} height={22} />
            </span>
           
          </button>
        )}
      </div>
    </form>
  )
}
